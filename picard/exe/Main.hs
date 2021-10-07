{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PartialTypeSignatures #-}

module Main (main, testServer) where

import Control.Applicative (Alternative (empty, (<|>)), optional)
import Control.Concurrent (threadDelay)
import Control.Concurrent.Async (mapConcurrently)
import qualified Control.Concurrent.MSem as MSem
import Control.Concurrent.STM.TVar (TVar, modifyTVar, newTVar, readTVar, readTVarIO, writeTVar)
import Control.Exception (throw)
import Control.Monad (ap, forever, void)
import Control.Monad.Except (MonadError (throwError), runExceptT)
import Control.Monad.IO.Class (MonadIO (liftIO))
import Control.Monad.Reader (runReaderT)
import Control.Monad.STM (STM, atomically, throwSTM)
import Control.Monad.State.Strict (MonadState (get), evalStateT, modify)
import Control.Monad.Trans (MonadTrans (lift))
import qualified Data.Attoparsec.Text as Atto (IResult (..), Parser, Result, char, endOfInput, feed, many', parse, skipSpace, string)
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy as LBS
import Data.Foldable (Foldable (foldl'))
import Data.Function (on)
import Data.Functor (($>), (<&>))
import qualified Data.HashMap.Strict as HashMap
import Data.List (sortBy)
import Data.Maybe (catMaybes, fromMaybe)
import qualified Data.Text as Text (Text, empty, length, pack, stripPrefix, unpack)
import qualified Data.Text.Encoding as Text
import Language.SQL.SpiderSQL.Lexer (lexSpiderSQL)
import Language.SQL.SpiderSQL.Parse (ParserEnv (..), ParserEnvWithGuards (..), mkParserStateTC, mkParserStateUD, spiderSQL, withGuards)
import Language.SQL.SpiderSQL.Prelude (caselessString)
import Language.SQL.SpiderSQL.Syntax (SX (..), SpiderSQLTC, SpiderSQLUD)
import qualified Network.HTTP.Client as HTTP
import qualified Network.HTTP.Client.TLS as HTTP
import qualified Picard.Picard.Client as Picard
import qualified Picard.Picard.Service as Picard
import qualified Picard.Types as Picard
import System.Timeout (timeout)
import qualified Thrift.Api as Thrift
import qualified Thrift.Channel.HeaderChannel as Thrift
import qualified Thrift.Protocol.Id as Thrift
import qualified Thrift.Server.CppServer as Thrift
import qualified Tokenizers (Tokenizer, createTokenizerFromJSONConfig, decode, freeTokenizer)
import Util.Control.Exception (catchAll)
import qualified Util.EventBase as Thrift

trace :: forall a. String -> a -> a
trace _ = id

data PartialParse a
  = PartialParse !Text.Text !(Atto.Result a)
  deriving stock (Show)

data PicardState = PicardState
  { counter :: TVar Int,
    sqlSchemas :: TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema),
    tokenizer :: TVar (Maybe Tokenizers.Tokenizer),
    partialSpiderSQLParsesWithGuardsAndTypeChecking :: TVar (HashMap.HashMap Picard.InputIds (PartialParse SpiderSQLTC)),
    partialSpiderSQLParsesWithGuards :: TVar (HashMap.HashMap Picard.InputIds (PartialParse SpiderSQLUD)),
    partialSpiderSQLParsesWithoutGuards :: TVar (HashMap.HashMap Picard.InputIds (PartialParse SpiderSQLUD)),
    partialSpiderSQLLexes :: TVar (HashMap.HashMap Picard.InputIds (PartialParse [String]))
  }

initPicardState :: IO PicardState
initPicardState =
  atomically $
    PicardState
      <$> newTVar 0
        <*> newTVar mempty
        <*> newTVar Nothing
        <*> newTVar mempty
        <*> newTVar mempty
        <*> newTVar mempty
        <*> newTVar mempty

mkSchemaParser ::
  HashMap.HashMap Picard.DBId Picard.SQLSchema ->
  Atto.Parser Picard.SQLSchema
mkSchemaParser sqlSchemas =
  foldl'
    (\agg (dbId, schema) -> agg <|> caselessString (Text.unpack dbId) $> schema)
    empty
    (sortBy (compare `on` (negate . Text.length . fst)) (HashMap.toList sqlSchemas))

mkParser ::
  forall a.
  Atto.Parser Picard.SQLSchema ->
  (Picard.SQLSchema -> Atto.Parser a) ->
  Atto.Parser a
mkParser schemaParser mkMainParser = do
  _ <-
    Atto.skipSpace
      *> Atto.many'
        ( Atto.char '<'
            *> (Atto.string "pad" <|> Atto.string "s" <|> Atto.string "/s")
            <* Atto.char '>'
        )
        <* Atto.skipSpace
  schema <- schemaParser
  _ <- Atto.skipSpace *> Atto.char '|' <* Atto.skipSpace
  mkMainParser schema
    <* optional (Atto.skipSpace <* Atto.char ';')
    <* Atto.endOfInput

getPartialParse ::
  forall a.
  HashMap.HashMap Picard.DBId Picard.SQLSchema ->
  (Picard.SQLSchema -> Atto.Parser a) ->
  Text.Text ->
  PartialParse a
getPartialParse sqlSchemas mkMainParser =
  let schemaParser = mkSchemaParser sqlSchemas
      m = mkParser schemaParser mkMainParser
   in ap PartialParse $ Atto.parse m

initializeParserCacheSTM ::
  forall a.
  (Picard.SQLSchema -> Atto.Parser a) ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse a)) ->
  STM ()
initializeParserCacheSTM mainParser sqlSchemas partialParses = do
  nukeParserCache partialParses
  partialParse <-
    getPartialParse
      <$> readTVar sqlSchemas
      <*> pure mainParser
      <*> pure mempty
  modifyTVar partialParses (HashMap.insert mempty partialParse)

nukeParserCache ::
  forall a.
  TVar (HashMap.HashMap Picard.InputIds (PartialParse a)) ->
  STM ()
nukeParserCache partialParses = writeTVar partialParses HashMap.empty

data LookupResult a = Cached !a | Fresh !a
  deriving stock (Show)

lookupResultIO ::
  forall a.
  HashMap.HashMap Picard.DBId Picard.SQLSchema ->
  (Picard.SQLSchema -> Atto.Parser a) ->
  (Picard.InputIds -> IO String) ->
  HashMap.HashMap Picard.InputIds (PartialParse a) ->
  Picard.InputIds ->
  IO (LookupResult (PartialParse a))
lookupResultIO sqlSchemas mkMainParser decode partialParses inputIds =
  case HashMap.lookup inputIds partialParses of
    Just partialParse ->
      trace ("Server: Found inputIds " <> show inputIds) . pure $ Cached partialParse
    Nothing ->
      trace ("Server: Did not find inputIds " <> show inputIds) $ do
        decodedInputIds <- decode inputIds
        let !partialParse = getPartialParse sqlSchemas mkMainParser (Text.pack decodedInputIds)
        pure $ Fresh partialParse

lookupResultWithTimeoutIO ::
  forall a m.
  (MonadState DebugInfo m, MonadError Picard.FeedTimeoutFailure m, MonadIO m) =>
  Int ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  (Picard.SQLSchema -> Atto.Parser a) ->
  (Picard.InputIds -> IO String) ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse a)) ->
  Picard.InputIds ->
  m (PartialParse a)
lookupResultWithTimeoutIO microSeconds sqlSchemas mkMainParser decode partialParses inputIds = do
  r <-
    resultOrTimeout
      microSeconds
      $ do
        schemas <- readTVarIO sqlSchemas
        parses <- readTVarIO partialParses
        !lr <- lookupResultIO schemas mkMainParser decode parses inputIds
        pure lr
  let f (Cached partialParse) = pure partialParse
      f (Fresh partialParse) =
        trace ("Server: Cached inputIds " <> show inputIds) . liftIO . atomically $ do
          modifyTVar partialParses $ HashMap.insert inputIds partialParse
          pure partialParse
      g partialParse@(PartialParse decodedInputIds _) = do
        modify (\debugInfo -> debugInfo {debugDecodedInputIds = Just decodedInputIds})
        pure partialParse
  pure r >>= f >>= g

decodedTokenFromDifferenceIO ::
  (Picard.InputIds -> IO String) ->
  Picard.InputIds ->
  Picard.Token ->
  Text.Text ->
  IO (Text.Text, Maybe Text.Text)
decodedTokenFromDifferenceIO decode inputIds token decodedInputIds = do
  decoded <- Text.pack <$> decode (inputIds ++ [token])
  pure (decoded, Text.stripPrefix decodedInputIds decoded)

decodedTokenFromDifferenceM ::
  forall m.
  (MonadState DebugInfo m, MonadIO m) =>
  (Picard.InputIds -> IO String) ->
  Picard.InputIds ->
  Picard.Token ->
  Text.Text ->
  m (Text.Text, Text.Text)
decodedTokenFromDifferenceM decode inputIds token decodedInputIds = do
  (decoded, maybeDecodedToken) <- liftIO $ decodedTokenFromDifferenceIO decode inputIds token decodedInputIds
  _ <- modify (\debugInfo -> debugInfo {debugDecoded = Just decoded})
  maybe
    ( trace
        ("Server: Prefix error " <> show decodedInputIds <> " " <> show decoded)
        . throw
        . Picard.FeedException
        . Picard.FeedFatalException_tokenizerPrefixException
        . Picard.TokenizerPrefixException
        $ "Prefix error."
    )
    ( \decodedToken -> do
        _ <- modify (\debugInfo -> debugInfo {debugDecodedToken = Just decodedToken})
        pure (decoded, decodedToken)
    )
    maybeDecodedToken

data DebugInfo = DebugInfo
  { debugInputIds :: Maybe Picard.InputIds,
    debugToken :: Maybe Picard.Token,
    debugDecodedInputIds :: Maybe Text.Text,
    debugDecodedToken :: Maybe Text.Text,
    debugDecoded :: Maybe Text.Text
  }

mkDebugInfo :: DebugInfo
mkDebugInfo = DebugInfo Nothing Nothing Nothing Nothing Nothing

resultOrTimeout :: forall r m. (MonadIO m, MonadState DebugInfo m, MonadError Picard.FeedTimeoutFailure m) => Int -> IO r -> m r
resultOrTimeout microSeconds ior = do
  mr <- liftIO $ timeout microSeconds ior
  case mr of
    Just r -> pure r
    Nothing -> do
      DebugInfo {..} <- get
      trace
        ("Server: Timeout error " <> show debugDecodedInputIds <> " " <> show debugDecoded)
        . throwError
        . Picard.FeedTimeoutFailure
        $ "Timeout error."

feedParserWithTimeoutIO ::
  forall a m.
  (MonadState DebugInfo m, MonadIO m, MonadError Picard.FeedTimeoutFailure m) =>
  Int ->
  Atto.Result a ->
  Text.Text ->
  m (Atto.Result a)
feedParserWithTimeoutIO microSeconds partialParseResult decodedToken = do
  resultOrTimeout
    microSeconds
    $ do
      let !r =
            Atto.feed
              partialParseResult
              ( case decodedToken of
                  "</s>" -> Text.empty
                  s -> s
              )
      pure r

-- | fix me: need one counter for each hashmap
nukeParserCacheEverySTM ::
  forall a.
  Int ->
  TVar Int ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse a)) ->
  STM ()
nukeParserCacheEverySTM n counter partialParses = do
  _ <- modifyTVar counter (+ 1)
  c <- readTVar counter
  case c `mod` n of
    0 -> nukeParserCache partialParses
    _ -> pure ()

toResult ::
  forall a m.
  (MonadState DebugInfo m, Show a) =>
  Atto.Result a ->
  m Picard.FeedResult
toResult (Atto.Done notConsumed _) = pure . Picard.FeedResult_feedCompleteSuccess . Picard.FeedCompleteSuccess $ notConsumed
toResult result@(Atto.Partial _) = do
  DebugInfo {..} <- get
  trace
    ( "Server: "
        <> "Partial. Input ids were: "
        <> show debugInputIds
        <> ". Token was: "
        <> show debugToken
        <> ". Decoded input ids were: "
        <> show debugDecodedInputIds
        <> ". Decoded token was: "
        <> show debugDecodedToken
        <> ". Result: "
        <> show result
        <> "."
    )
    . pure
    . Picard.FeedResult_feedPartialSuccess
    $ Picard.FeedPartialSuccess
toResult result@(Atto.Fail i contexts description) = do
  DebugInfo {..} <- get
  trace
    ( "Server: "
        <> "Failure. Input ids were: "
        <> show debugInputIds
        <> ". Token was: "
        <> show debugToken
        <> ". Decoded input ids were: "
        <> show debugDecodedInputIds
        <> ". Decoded token was: "
        <> show debugDecodedToken
        <> ". Result: "
        <> show result
        <> "."
    )
    . pure
    . Picard.FeedResult_feedParseFailure
    $ Picard.FeedParseFailure
      { feedParseFailure_input = i,
        feedParseFailure_contexts = Text.pack <$> contexts,
        feedParseFailure_description = Text.pack description
      }

-- | fix me: the tokenizer referenced here may be freed and/or replaced elsewhere...
getDecode :: TVar (Maybe Tokenizers.Tokenizer) -> IO (Picard.InputIds -> IO String)
getDecode maybeTokenizer =
  do
    tok <-
      fromMaybe
        ( throw
            . Picard.FeedException
            . Picard.FeedFatalException_tokenizerNotRegisteredException
            . Picard.TokenizerNotRegisteredException
            $ "Tokenizer has not been registered."
        )
        <$> readTVarIO maybeTokenizer
    tokSem <- MSem.new (1 :: Int)
    pure (\inputIds -> MSem.with tokSem (Tokenizers.decode tok $ fromIntegral <$> inputIds))

feedIO ::
  forall a.
  Show a =>
  TVar Int ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  (Picard.SQLSchema -> Atto.Parser a) ->
  TVar (Maybe Tokenizers.Tokenizer) ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse a)) ->
  Picard.InputIds ->
  Picard.Token ->
  IO Picard.FeedResult
feedIO counter sqlSchemas mkMainParser maybeTokenizer partialParses inputIds token =
  evalStateT
    ( runExceptT
        ( do
            _ <- liftIO . atomically $ nukeParserCacheEverySTM 10000 counter partialParses
            decode <- liftIO . getDecode $ maybeTokenizer
            partialParse <- getPartialParseIO decode
            liftIO . atomically . modifyTVar partialParses $ HashMap.insert (inputIds ++ [token]) partialParse
            pure partialParse
        )
        >>= toResultIO
    )
    initialDebugInfo
  where
    initialDebugInfo =
      mkDebugInfo
        { debugInputIds = Just inputIds,
          debugToken = Just token
        }
    getPartialParseIO tokenizer = do
      let microSeconds = 100000
      PartialParse decodedInputIds partialParseResult <-
        lookupResultWithTimeoutIO microSeconds sqlSchemas mkMainParser tokenizer partialParses inputIds
      (decoded, decodedToken) <-
        decodedTokenFromDifferenceM tokenizer inputIds token decodedInputIds
      partialParseResult' <-
        feedParserWithTimeoutIO microSeconds partialParseResult decodedToken
      pure $ PartialParse decoded partialParseResult'
    toResultIO (Left timeoutFailure) = pure $ Picard.FeedResult_feedTimeoutFailure timeoutFailure
    toResultIO (Right (PartialParse _ r)) = toResult r

batchFeedIO ::
  forall a.
  Show a =>
  TVar Int ->
  TVar (HashMap.HashMap Picard.DBId Picard.SQLSchema) ->
  (Picard.SQLSchema -> Atto.Parser a) ->
  TVar (Maybe Tokenizers.Tokenizer) ->
  TVar (HashMap.HashMap Picard.InputIds (PartialParse a)) ->
  [Picard.InputIds] ->
  [[Picard.Token]] ->
  IO [Picard.BatchFeedResult]
batchFeedIO counter sqlSchemas mkMainParser maybeTokenizer partialParses inputIds topTokens =
  do
    decode <- getDecode maybeTokenizer
    results <-
      mapConcurrently
        (action decode)
        . concat
        . zipWith3
          (\batchId inputIds' tokens -> (batchId,inputIds',) <$> tokens)
          [0 :: Picard.BatchId ..]
          inputIds
        $ topTokens
    let newPartialParses = HashMap.fromList $ catMaybes (fst <$> results)
    atomically . modifyTVar partialParses $ HashMap.union newPartialParses
    pure $ snd <$> results
  where
    action :: (Picard.InputIds -> IO String) -> (Picard.BatchId, Picard.InputIds, Picard.Token) -> IO (Maybe ([Picard.Token], PartialParse a), Picard.BatchFeedResult)
    action decode (batchId, inputIds', token) =
      evalStateT
        ( runExceptT
            ( do
                _ <- liftIO . atomically $ nukeParserCacheEverySTM 10000 counter partialParses
                getPartialParseIO
            )
            >>= ( \partialParse ->
                    (either (const Nothing) (Just . (inputIds' ++ [token],)) partialParse,)
                      <$> ( Picard.BatchFeedResult batchId token <$> toResultIO partialParse
                          )
                )
        )
        initialDebugInfo
      where
        initialDebugInfo =
          mkDebugInfo
            { debugInputIds = Just inputIds',
              debugToken = Just token
            }
        getPartialParseIO ::
          forall m.
          (MonadError Picard.FeedTimeoutFailure m, MonadState DebugInfo m, MonadIO m) =>
          m (PartialParse a)
        getPartialParseIO = do
          let microSeconds = 100000
          PartialParse decodedInputIds partialParseResult <-
            lookupResultWithTimeoutIO microSeconds sqlSchemas mkMainParser decode partialParses inputIds'
          (decoded, decodedToken) <-
            decodedTokenFromDifferenceM decode inputIds' token decodedInputIds
          partialParseResult' <-
            feedParserWithTimeoutIO microSeconds partialParseResult decodedToken
          pure $ PartialParse decoded partialParseResult'
        toResultIO (Left timeoutFailure) = pure $ Picard.FeedResult_feedTimeoutFailure timeoutFailure
        toResultIO (Right (PartialParse _ r)) = toResult r

batchFeedIO' counter sqlSchemas mkMainParser tokenizer partialParses inputIds topTokens =
  do
    decode <- getDecode tokenizer
    schemas <- readTVarIO sqlSchemas
    -- traverse
    -- mapPool (length inputIds)
    mapConcurrently
      ( \(batchId, inputIds', token) ->
          evalStateT
            ( runExceptT
                ( do
                    _ <- liftIO . atomically $ nukeParserCacheEverySTM 10000 counter partialParses
                    let microSeconds = 10000000
                    PartialParse decodedInputIds' partialParseResult <-
                      resultOrTimeout
                        microSeconds
                        ( do
                            parses <- readTVarIO partialParses
                            !lr <- lookupResultIO schemas mkMainParser decode parses inputIds'
                            pure lr
                        )
                        >>= ( \case
                                Cached partialParse -> pure partialParse
                                Fresh partialParse -> trace ("Server: Cached inputIds " <> show inputIds') . liftIO . atomically $ do
                                  modifyTVar partialParses $ HashMap.insert inputIds' partialParse
                                  pure partialParse
                            )
                    (decoded, decodedToken) <- decodedTokenFromDifferenceM decode inputIds' token decodedInputIds'
                    modify (\debugInfo -> debugInfo {debugDecodedInputIds = Just decodedInputIds'})
                    partialParseResult' <-
                      resultOrTimeout
                        microSeconds
                        $ let !r =
                                Atto.feed
                                  partialParseResult
                                  ( case decodedToken of
                                      "</s>" -> Text.empty
                                      s -> s
                                  )
                           in pure r
                    let partialParse = PartialParse decoded partialParseResult'
                    liftIO . atomically . modifyTVar partialParses $ HashMap.insert (inputIds' ++ [token]) partialParse
                    pure partialParse
                )
                >>= ( \case
                        Left timeoutFailure -> pure $ Picard.FeedResult_feedTimeoutFailure timeoutFailure
                        Right (PartialParse _ r) -> toResult r
                    )
                <&> Picard.BatchFeedResult batchId token
            )
            ( mkDebugInfo
                { debugInputIds = Just inputIds',
                  debugToken = Just token
                }
            )
      )
      . concat
      . zipWith3
        (\batchId inputIds' tokens -> (batchId,inputIds',) <$> tokens)
        [0 :: Picard.BatchId ..]
        inputIds
      $ topTokens

mapPool :: forall a b t. Traversable t => Int -> (a -> IO b) -> t a -> IO (t b)
mapPool size f xs = do
  sem <- MSem.new size
  mapConcurrently (MSem.with sem . f) xs

picardHandler :: forall a. PicardState -> Picard.PicardCommand a -> IO a
picardHandler PicardState {..} = go
  where
    mkSpiderSQLParserWithGuardsAndTypeChecking :: Picard.SQLSchema -> Atto.Parser SpiderSQLTC
    mkSpiderSQLParserWithGuardsAndTypeChecking = runReaderT (spiderSQL STC mkParserStateTC) . ParserEnv (ParserEnvWithGuards (withGuards STC))
    mkSpiderSQLParserWithGuards :: Picard.SQLSchema -> Atto.Parser SpiderSQLUD
    mkSpiderSQLParserWithGuards = runReaderT (spiderSQL SUD mkParserStateUD) . ParserEnv (ParserEnvWithGuards (withGuards SUD))
    mkSpiderSQLParserWithoutGuards :: Picard.SQLSchema -> Atto.Parser SpiderSQLUD
    mkSpiderSQLParserWithoutGuards = runReaderT (spiderSQL SUD mkParserStateUD) . ParserEnv (ParserEnvWithGuards (const id))
    mkSpiderSQLLexer :: Picard.SQLSchema -> Atto.Parser [String]
    mkSpiderSQLLexer = runReaderT lexSpiderSQL
    go (Picard.RegisterSQLSchema dbId sqlSchema) =
      trace ("RegisterSQLSchema " <> show dbId) $
        atomically $ do
          r <- readTVar sqlSchemas
          case HashMap.lookup dbId r of
            Just _ -> throwSTM $ Picard.RegisterSQLSchemaException dbId "Database schema is already registered"
            Nothing -> do
              modifyTVar sqlSchemas (HashMap.insert dbId sqlSchema)
              initializeParserCacheSTM mkSpiderSQLParserWithGuardsAndTypeChecking sqlSchemas partialSpiderSQLParsesWithGuardsAndTypeChecking
              initializeParserCacheSTM mkSpiderSQLParserWithGuards sqlSchemas partialSpiderSQLParsesWithGuards
              initializeParserCacheSTM mkSpiderSQLParserWithoutGuards sqlSchemas partialSpiderSQLParsesWithoutGuards
              initializeParserCacheSTM mkSpiderSQLLexer sqlSchemas partialSpiderSQLLexes
    go (Picard.RegisterTokenizer jsonConfig) =
      trace "RegisterTokenizer" $ do
        tok <- Tokenizers.createTokenizerFromJSONConfig . Text.encodeUtf8 $ jsonConfig
        maybeOldTokenizer <- atomically $ do
          maybeOldTokenizer <- readTVar tokenizer
          writeTVar tokenizer . Just $ tok
          initializeParserCacheSTM mkSpiderSQLParserWithGuardsAndTypeChecking sqlSchemas partialSpiderSQLParsesWithGuardsAndTypeChecking
          initializeParserCacheSTM mkSpiderSQLParserWithGuards sqlSchemas partialSpiderSQLParsesWithGuards
          initializeParserCacheSTM mkSpiderSQLParserWithoutGuards sqlSchemas partialSpiderSQLParsesWithoutGuards
          initializeParserCacheSTM mkSpiderSQLLexer sqlSchemas partialSpiderSQLLexes
          pure maybeOldTokenizer
        case maybeOldTokenizer of
          Just oldTok -> Tokenizers.freeTokenizer oldTok
          Nothing -> pure ()
    go (Picard.Feed inputIds token Picard.Mode_PARSING_WITH_GUARDS_AND_TYPE_CHECKING) =
      trace ("Feed parsing with guards " <> show inputIds <> " " <> show token) $
        feedIO counter sqlSchemas mkSpiderSQLParserWithGuardsAndTypeChecking tokenizer partialSpiderSQLParsesWithGuardsAndTypeChecking inputIds token
    go (Picard.Feed inputIds token Picard.Mode_PARSING_WITH_GUARDS) =
      trace ("Feed parsing with guards " <> show inputIds <> " " <> show token) $
        feedIO counter sqlSchemas mkSpiderSQLParserWithGuards tokenizer partialSpiderSQLParsesWithGuards inputIds token
    go (Picard.Feed inputIds token Picard.Mode_PARSING_WITHOUT_GUARDS) =
      trace ("Feed parsing without guards " <> show inputIds <> " " <> show token) $
        feedIO counter sqlSchemas mkSpiderSQLParserWithoutGuards tokenizer partialSpiderSQLParsesWithoutGuards inputIds token
    go (Picard.Feed inputIds token Picard.Mode_LEXING) =
      trace ("Feed lexing " <> show inputIds <> " " <> show token) $
        feedIO counter sqlSchemas mkSpiderSQLLexer tokenizer partialSpiderSQLLexes inputIds token
    go (Picard.Feed _inputIds _token (Picard.Mode__UNKNOWN n)) =
      throw
        . Picard.FeedException
        . Picard.FeedFatalException_modeException
        . Picard.ModeException
        . Text.pack
        $ "Unknown mode " <> show n
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_PARSING_WITH_GUARDS_AND_TYPE_CHECKING) =
      batchFeedIO' counter sqlSchemas mkSpiderSQLParserWithGuardsAndTypeChecking tokenizer partialSpiderSQLParsesWithGuardsAndTypeChecking inputIds topTokens
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_PARSING_WITH_GUARDS) =
      batchFeedIO' counter sqlSchemas mkSpiderSQLParserWithGuards tokenizer partialSpiderSQLParsesWithGuards inputIds topTokens
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_PARSING_WITHOUT_GUARDS) =
      batchFeedIO' counter sqlSchemas mkSpiderSQLParserWithoutGuards tokenizer partialSpiderSQLParsesWithoutGuards inputIds topTokens
    go (Picard.BatchFeed inputIds topTokens Picard.Mode_LEXING) =
      batchFeedIO' counter sqlSchemas mkSpiderSQLLexer tokenizer partialSpiderSQLLexes inputIds topTokens
    go (Picard.BatchFeed _inputIds _token (Picard.Mode__UNKNOWN n)) =
      throw
        . Picard.FeedException
        . Picard.FeedFatalException_modeException
        . Picard.ModeException
        . Text.pack
        $ "Unknown mode " <> show n

withPicardServer :: forall a. Thrift.ServerOptions -> (Int -> IO a) -> IO a
withPicardServer serverOptions action = do
  st <- initPicardState
  Thrift.withBackgroundServer (picardHandler st) serverOptions $
    \Thrift.Server {..} -> action serverPort

picardServerHost :: BS8.ByteString
picardServerHost = BS8.pack "127.0.0.1"

mkHeaderConfig :: forall t. Int -> Thrift.ProtocolId -> Thrift.HeaderConfig t
mkHeaderConfig port protId =
  Thrift.HeaderConfig
    { headerHost = picardServerHost,
      headerPort = port,
      headerProtocolId = protId,
      headerConnTimeout = 5000,
      headerSendTimeout = 5000,
      headerRecvTimeout = 5000
    }

testServer :: IO ()
testServer = do
  let protId = Thrift.binaryProtocolId
      action :: Thrift.Thrift Picard.Picard ()
      action = do
        Picard.registerSQLSchema "test" $
          let columnNames = HashMap.fromList [("0", "column")]
              columnTypes = HashMap.fromList [("0", Picard.ColumnType_NUMBER)]
              tableNames = HashMap.fromList [("0", "table")]
              columnToTable = HashMap.fromList [("0", "0")]
              tableToColumns = HashMap.fromList [("0", ["0"])]
              foreignKeys = HashMap.fromList []
              primaryKeys = []
           in Picard.SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}
        Picard.registerSQLSchema "car_1" $
          let columnNames = HashMap.fromList [("1", "ContId"), ("10", "ModelId"), ("11", "Maker"), ("12", "Model"), ("13", "MakeId"), ("14", "Model"), ("15", "Make"), ("16", "Id"), ("17", "MPG"), ("18", "Cylinders"), ("19", "Edispl"), ("2", "Continent"), ("20", "Horsepower"), ("21", "Weight"), ("22", "Accelerate"), ("23", "Year"), ("3", "CountryId"), ("4", "CountryName"), ("5", "Continent"), ("6", "Id"), ("7", "Maker"), ("8", "FullName"), ("9", "Country")]
              columnTypes = HashMap.fromList [("1", Picard.ColumnType_NUMBER), ("10", Picard.ColumnType_NUMBER), ("11", Picard.ColumnType_NUMBER), ("12", Picard.ColumnType_TEXT), ("13", Picard.ColumnType_NUMBER), ("14", Picard.ColumnType_TEXT), ("15", Picard.ColumnType_TEXT), ("16", Picard.ColumnType_NUMBER), ("17", Picard.ColumnType_TEXT), ("18", Picard.ColumnType_NUMBER), ("19", Picard.ColumnType_NUMBER), ("2", Picard.ColumnType_TEXT), ("20", Picard.ColumnType_TEXT), ("21", Picard.ColumnType_NUMBER), ("22", Picard.ColumnType_NUMBER), ("23", Picard.ColumnType_NUMBER), ("3", Picard.ColumnType_NUMBER), ("4", Picard.ColumnType_TEXT), ("5", Picard.ColumnType_NUMBER), ("6", Picard.ColumnType_NUMBER), ("7", Picard.ColumnType_TEXT), ("8", Picard.ColumnType_TEXT), ("9", Picard.ColumnType_TEXT)]
              tableNames = HashMap.fromList [("0", "continents"), ("1", "countries"), ("2", "car_makers"), ("3", "model_list"), ("4", "car_names"), ("5", "cars_data")]
              columnToTable = HashMap.fromList [("1", "0"), ("10", "3"), ("11", "3"), ("12", "3"), ("13", "4"), ("14", "4"), ("15", "4"), ("16", "5"), ("17", "5"), ("18", "5"), ("19", "5"), ("2", "0"), ("20", "5"), ("21", "5"), ("22", "5"), ("23", "5"), ("3", "1"), ("4", "1"), ("5", "1"), ("6", "2"), ("7", "2"), ("8", "2"), ("9", "2")]
              tableToColumns = HashMap.fromList [("0", ["1", "2"]), ("1", ["3", "4", "5"]), ("2", ["6", "7", "8", "9"]), ("3", ["10", "11", "12"]), ("4", ["13", "14", "15"]), ("5", ["16", "17", "18", "19", "20", "21", "22", "23"])]
              foreignKeys = HashMap.fromList [("11", "6"), ("14", "12"), ("16", "13"), ("5", "1"), ("9", "3")]
              primaryKeys = ["1", "3", "6", "10", "13", "16"]
           in Picard.SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_columnTypes = columnTypes, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_primaryKeys = primaryKeys}
        manager <- lift $ HTTP.newTlsManagerWith HTTP.tlsManagerSettings
        request <- lift $ HTTP.parseRequest "https://huggingface.co/t5-base/resolve/main/tokenizer.json"
        response <- lift $ HTTP.httpLbs request manager
        Picard.registerTokenizer . Text.decodeUtf8 . LBS.toStrict $ HTTP.responseBody response
        let tokens = [0 .. 32100]
        mapM_
          ( flip catchAll (const (pure Nothing))
              . (Just <$>)
              . (\token -> Picard.feed [0, 794, 1820, 1738, 953, 5, 3297, 440, 29, 45, 953] token Picard.Mode_PARSING_WITH_GUARDS)
          )
          tokens
        let inputIds = [0, 443, 834, 536, 1820, 1738, 3, 17, 5411, 17529, 23, 26, 6, 3, 17, 5411, 17529, 4350, 45, 1440, 38, 3, 17, 536, 1715, 443, 834, 8910, 38, 3, 17, 357, 30, 3, 17, 5411, 17529, 23, 26, 3274, 3, 17, 4416, 17529, 563, 57, 3, 17, 5411, 17529, 23, 26, 578, 3476, 599, 1935, 61, 2490, 220, 7021, 1738, 3, 17, 5411, 17529, 23, 26, 6, 3, 17, 5411, 17529, 4350, 45, 1440, 38, 3, 17, 536, 1715, 443, 834, 8910, 38, 3, 17, 357, 30, 3, 17, 5411, 17529, 23, 26, 3274, 3, 17, 4416, 17529, 1715, 825, 834, 3350, 38, 3, 17, 519, 30, 3, 17, 4416, 23, 26, 3274, 3, 17, 5787, 8337, 213, 3, 17, 5787, 21770, 3274, 96, 3183, 144, 121, 1]
        void $
          Picard.feed
            (take 127 inputIds)
            (inputIds !! 127)
            Picard.Mode_PARSING_WITH_GUARDS
        void $
          Picard.feed
            (take 128 inputIds)
            (inputIds !! 128)
            Picard.Mode_PARSING_WITH_GUARDS
  withPicardServer Thrift.defaultOptions $
    \port ->
      Thrift.withEventBaseDataplane $ \evb -> do
        let headerConf = mkHeaderConfig port protId
        Thrift.withHeaderChannel evb headerConf action

main :: IO ()
main = do
  st <- initPicardState
  let serverOptions =
        Thrift.ServerOptions
          { desiredPort = Just 9090,
            customFactoryFn = Nothing,
            customModifyFn = Nothing
          }
      action Thrift.Server {} =
        forever (threadDelay maxBound)
  Thrift.withBackgroundServer (picardHandler st) serverOptions action
