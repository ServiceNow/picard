module Language.SQL.SpiderSQL.Parse where

import Control.Applicative (Alternative (..), Applicative (liftA2), optional)
import Control.Lens ((%~), (^.))
import Control.Monad (MonadPlus, forM_, join, unless, when)
import Control.Monad.Reader (runReaderT)
import Control.Monad.Reader.Class (MonadReader (ask))
import Control.Monad.State.Class (MonadState (..), modify)
import Control.Monad.State.Strict (evalStateT)
import Data.Char (isAlpha, isAlphaNum, toLower)
import Data.Foldable (Foldable (foldl'))
import Data.Functor (($>))
import Data.Generics.Product (field)
import qualified Data.HashMap.Strict as HashMap (HashMap, compose, filter, insertWith, keys, lookup, member, toList)
import qualified Data.HashSet as HashSet
import Data.Hashable (Hashable)
import qualified Data.Map as Map (Map, fromListWith, lookupLE, member, singleton, toList, union, unionWith)
import Data.Maybe (catMaybes, fromMaybe)
import qualified Data.Text as Text
import Data.Word (Word8)
import GHC.Generics (Generic)
import Language.SQL.SpiderSQL.Prelude (columnNameP, doubleP, eitherP, intP, isAnd, isAs, isAsc, isAvg, isBetween, isClosedParenthesis, isComma, isCount, isDesc, isDistinct, isDivide, isDot, isEq, isExcept, isFrom, isGe, isGroupBy, isGt, isHaving, isIn, isIntersect, isJoin, isLe, isLike, isLimit, isLt, isMax, isMin, isMinus, isNe, isNot, isOn, isOpenParenthesis, isOr, isOrderBy, isPlus, isSelect, isStar, isSum, isTimes, isUnion, isWhere, manyAtMost, quotedString, tableNameP)
import Language.SQL.SpiderSQL.Syntax (Agg (..), AggType (..), Alias (..), ColUnit (..), ColumnId (..), Cond (..), From (..), OrderBy (..), OrderByOrder (..), Select (..), SpiderSQL (..), TableId (..), TableUnit (..), Val (..), ValUnit (..))
import Picard.Types (SQLSchema (..))
import Text.Parser.Char (CharParsing (..), alphaNum, digit, spaces)
import Text.Parser.Combinators (Parsing (..), between, choice, sepBy, sepBy1)
import Text.Parser.Permutation (permute, (<$$>), (<||>))
import Text.Parser.Token (TokenParsing (..))

-- $setup
-- >>> :set -XOverloadedStrings
-- >>> import qualified Data.Attoparsec.Text as Atto (parse, parseOnly, endOfInput, string, char)
-- >>> import Picard.Types (SQLSchema)
-- >>> import Control.Monad.Reader (runReader, runReaderT)
-- >>> import Control.Monad.Trans (MonadTrans (lift))
-- >>> import qualified Data.HashMap.Strict as HashMap
-- >>> columnNames = HashMap.fromList [("1", "Singer_ID"), ("2", "Name"), ("3", "Birth_Year"), ("4", "Net_Worth_Millions"), ("5", "Citizenship"), ("6", "Song_ID"), ("7", "Title"), ("8", "Singer_ID"), ("9", "Sales"), ("10", "Highest_Position")] :: HashMap.HashMap Text.Text Text.Text
-- >>> tableNames = HashMap.fromList [("0", "singer"), ("1", "song")] :: HashMap.HashMap Text.Text Text.Text
-- >>> columnToTable = HashMap.fromList [("1", "0"), ("2", "0"), ("3", "0"), ("4", "0"), ("5", "0"), ("6", "1"), ("7", "1"), ("8", "1"), ("9", "1"), ("10", "1")] :: HashMap.HashMap Text.Text Text.Text
-- >>> tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5"]), ("1", ["6", "7", "8", "9", "10"])] :: HashMap.HashMap Text.Text [Text.Text]
-- >>> foreignKeys = HashMap.fromList [("8", "1")] :: HashMap.HashMap Text.Text Text.Text
-- >>> foreignKeysTables = HashMap.fromList [("1", ["0"])] :: HashMap.HashMap Text.Text [Text.Text]
-- >>> primaryKeys = ["1", "6"] :: [Text.Text]
-- >>> sqlSchema = SQLSchema { sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys }
-- >>> parserEnv = ParserEnv (ParserEnvWithGuards withGuards) sqlSchema
-- >>> testParse = Atto.parse . flip runReaderT parserEnv . flip evalStateT mkParserState
-- >>> testParseOnly p = (Atto.parseOnly . flip runReaderT parserEnv . flip evalStateT mkParserState) (p <* (lift . lift) Atto.endOfInput)
-- >>> spiderSQLTestParse = (Atto.parse . flip runReaderT parserEnv) (spiderSQL mkParserState)
-- >>> spiderSQLTestParseOnly = (Atto.parseOnly . flip runReaderT parserEnv) (spiderSQL mkParserState <* lift Atto.endOfInput)

-- | ParserState
--
-- A table alias defined in scope n is:
-- - valid in scopes n, n + 1, ... unless shadowed by an alias defined in scope n' > n,
-- - not valid in scopes 0, 1, ..., n - 1.
data ParserState = ParserState
  { psAliases :: HashMap.HashMap Alias (Map.Map Scope TableUnit),
    psTables :: HashMap.HashMap (Either TableId Select) (HashSet.HashSet Scope),
    psCurScope :: Scope,
    psGuards :: HashMap.HashMap Scope (HashSet.HashSet Guard)
  }
  deriving stock (Eq, Show, Generic)

newtype Scope = Scope Word8
  deriving stock (Generic)
  deriving (Show, Eq, Ord, Num, Enum, Bounded, Hashable) via Word8

data Guard
  = GuardTableColumn TableId ColumnId
  | GuardAliasColumn Alias ColumnId
  | GuardColumn ColumnId
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (Hashable)

mkParserState :: ParserState
mkParserState = ParserState {psAliases = mempty, psTables = mempty, psCurScope = minBound, psGuards = mempty}

newtype ParserEnvWithGuards
  = ParserEnvWithGuards
      ( forall m p.
        ( Parsing m,
          MonadPlus m,
          MonadState ParserState m
        ) =>
        SQLSchema ->
        m p ->
        m p
      )

-- | ParserEnv
data ParserEnv = ParserEnv
  { peWithGuards :: ParserEnvWithGuards,
    peSQLSchema :: SQLSchema
  }
  deriving stock (Generic)

type MonadSQL m = (MonadPlus m, MonadState ParserState m, MonadReader ParserEnv m)

-- >>> testParseOnly (betweenParentheses $ char 'x') "x"
-- Left "\"(\": satisfyElem"
--
-- >>> testParseOnly (betweenParentheses $ char 'x') "(x)"
-- Right 'x'
--
-- >>> testParseOnly (betweenParentheses $ char 'x') "( x )"
-- Right 'x'
betweenParentheses :: CharParsing m => m a -> m a
betweenParentheses =
  between
    (isOpenParenthesis <* spaces)
    (spaces *> isClosedParenthesis)

-- >>> testParseOnly (betweenOptionalParentheses $ char 'x') "x"
-- Right 'x'
--
-- >>> testParseOnly (betweenOptionalParentheses $ char 'x') "(x)"
-- Right 'x'
betweenOptionalParentheses :: CharParsing m => m a -> m a
betweenOptionalParentheses p = betweenParentheses p <|> p

-- | 'Select' parser
--
-- >>> testParseOnly select "select *"
-- Right (Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))])
--
-- >>> testParseOnly select "select count singer.*"
-- Right (Select [Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Left (TableId {tableName = "singer"})), colUnitColId = Star}}))])
--
-- >>> testParseOnly select "SELECT COUNT (DISTINCT song.Title)"
-- Right (Select [Agg (Just Count) (Column (ValColUnit {columnValue = DistinctColUnit {distinctColUnitAggId = Nothing, distinctColUnitTable = Just (Left (TableId {tableName = "song"})), distinctColUnitColdId = ColumnId {columnName = "Title"}}}))])
--
-- >>> testParseOnly select "SELECT COUNT (DISTINCT T1.Title)"
-- Right (Select [Agg (Just Count) (Column (ValColUnit {columnValue = DistinctColUnit {distinctColUnitAggId = Nothing, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = ColumnId {columnName = "Title"}}}))])
select :: (TokenParsing m, MonadSQL m) => m Select
select = flip (<?>) "select" $ do
  _ <- isSelect
  someSpace
  distinct <- optional (isDistinct <* spaces)
  aggs <- sepBy (betweenOptionalParentheses agg) (spaces *> isComma <* spaces)
  case distinct of
    Just _ -> pure $ SelectDistinct aggs
    Nothing -> pure $ Select aggs

-- | 'Agg' parser.
--
-- >>> testParseOnly agg "singer.Singer_ID"
-- Right (Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Left (TableId {tableName = "singer"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly agg "count *"
-- Right (Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}})))
--
-- >>> testParseOnly agg "count (*)"
-- Right (Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}})))
--
-- >>> testParseOnly agg "count(*)"
-- Right (Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}})))
--
-- >>> testParseOnly agg "count singer.Singer_ID"
-- Right (Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Left (TableId {tableName = "singer"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))
agg :: (TokenParsing m, MonadSQL m) => m Agg
agg =
  flip (<?>) "agg" $ do
    at <- optional aggType
    Agg at
      <$> case at of
        Nothing -> valUnit
        Just _ ->
          (spaces *> betweenParentheses (spaces *> valUnit <* spaces))
            <|> someSpace *> valUnit

-- | 'AggType' parser.
--
-- >>> testParseOnly aggType ""
-- Left "aggType: Failed reading: mzero"
--
-- >>> testParseOnly aggType "sum"
-- Right Sum
aggType :: CharParsing m => m AggType
aggType = flip (<?>) "aggType" $ choice choices
  where
    choices =
      [ isMax $> Max,
        isMin $> Min,
        isCount $> Count,
        isSum $> Sum,
        isAvg $> Avg
      ]

-- | 'ValUnit' parser.
--
-- >>> testParseOnly valUnit "t1.Singer_ID"
-- Right (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}))
--
-- >>> testParseOnly valUnit "t2.Sales / t1.Net_Worth_Millions"
-- Right (Divide (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Sales"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Net_Worth_Millions"}}}))
--
-- >>> testParseOnly valUnit "t2.Sales / 4"
-- Right (Divide (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Sales"}}}) (Number {numberValue = 4.0}))
valUnit :: forall m. (TokenParsing m, MonadSQL m) => m ValUnit
valUnit =
  flip (<?>) "valUnit" $ do
    column <- val
    maybeBinary <-
      optional
        ( someSpace
            *> choice
              [ isMinus $> Minus column,
                isPlus $> Plus column,
                isTimes $> Times column,
                isDivide $> Divide column
              ]
        )
    case maybeBinary of
      Nothing -> pure $ Column column
      Just binary -> binary <$> (someSpace *> val)

-- | 'ColUnit' parser.
--
-- >>> testParseOnly colUnit "*"
-- Right (ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly colUnit "Singer_ID"
-- Right (ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Singer_ID"}})
--
-- >>> testParseOnly colUnit "distinct Singer_ID"
-- Right (DistinctColUnit {distinctColUnitAggId = Nothing, distinctColUnitTable = Nothing, distinctColUnitColdId = ColumnId {columnName = "Singer_ID"}})
--
-- >>> testParseOnly colUnit "t1.Singer_ID"
-- Right (ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}})
--
-- >>> testParseOnly colUnit "count *"
-- Right (ColUnit {colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly colUnit "count (*)"
-- Right (ColUnit {colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly colUnit "count(*)"
-- Right (ColUnit {colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly colUnit "count ( * )"
-- Right (ColUnit {colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star})
--
-- >>> testParseOnly colUnit "count t1.Singer_ID"
-- Right (ColUnit {colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}})
--
-- >>> testParseOnly colUnit "count distinct t1.*"
-- Right (DistinctColUnit {distinctColUnitAggId = Just Count, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = Star})
--
-- >>> testParseOnly colUnit "count (distinct t1.*)"
-- Right (DistinctColUnit {distinctColUnitAggId = Just Count, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = Star})
--
-- >>> testParseOnly colUnit "count(distinct t1.*)"
-- Right (DistinctColUnit {distinctColUnitAggId = Just Count, distinctColUnitTable = Just (Right (Alias {aliasName = "T1"})), distinctColUnitColdId = Star})
--
-- >>> (Atto.parseOnly . flip runReaderT (ParserEnv (ParserEnvWithGuards withGuards) sqlSchema { sQLSchema_columnNames = HashMap.union (sQLSchema_columnNames sqlSchema) (HashMap.singleton "11" "country") }) . flip evalStateT mkParserState) colUnit "country"
-- Right (ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "country"}})
colUnit ::
  forall m.
  ( TokenParsing m,
    MonadSQL m
  ) =>
  m ColUnit
colUnit = flip (<?>) "colUnit" $ do
  at <- optional aggType
  (distinct, tabAli, col) <- do
    let p = do
          distinct <- optional (isDistinct <* someSpace)
          (tabAli, col) <-
            ((Nothing,) <$> columnId)
              <|> ( (,)
                      <$> (Just <$> (eitherP tableId alias' <* isDot))
                        <*> columnId
                  )
          pure (distinct, tabAli, col)
    case at of
      Nothing -> p
      Just _ -> (spaces *> betweenParentheses p) <|> (someSpace *> p)
  v <-
    HashSet.singleton <$> case tabAli of
      Just (Left t) -> do
        ParserEnv _ sqlSchema <- ask
        columnInTable <- guardTableColumn sqlSchema t col
        case columnInTable of
          TableNotInScope -> pure $ GuardTableColumn t col
          ColumnNotInTable ->
            unexpected $
              "column " <> show col <> " is not in table " <> show t
          ColumnInTable -> pure $ GuardTableColumn t col
      Just (Right a) -> do
        ParserEnv _ sqlSchema <- ask
        columnInAlias <- guardAliasColumn sqlSchema a col
        case columnInAlias of
          AliasNotInScope -> pure $ GuardAliasColumn a col
          ColumnNotInAlias ->
            unexpected $
              "column " <> show col <> " is not in alias " <> show a
          ColumnInAlias -> pure $ GuardAliasColumn a col
      Nothing -> pure $ GuardColumn col
  curScope <- (^. field @"psCurScope") <$> get
  modify (field @"psGuards" %~ HashMap.insertWith HashSet.union curScope v)
  case distinct of
    Just _ -> pure $ DistinctColUnit at tabAli col
    Nothing -> pure $ ColUnit at tabAli col

-- | @inTable sqlSchema colId tabId@ checks if the 'ColumnId' @colId@ is valid for the table with the 'TableId' @tabId@ in the SQLSchema @sqlSchema@.
--
-- >>> inTable sqlSchema (ColumnId "Singer_ID") (TableId "song")
-- True
--
-- >>> inTable sqlSchema (ColumnId "singer_id") (TableId "song")
-- True
--
-- >>> inTable sqlSchema (ColumnId "Citizenship") (TableId "song")
-- False
inTable :: SQLSchema -> ColumnId -> TableId -> Bool
inTable _ Star _ = True
inTable SQLSchema {..} ColumnId {..} TableId {..} =
  let matchingColumnUIds =
        HashMap.keys
          . HashMap.filter (\x -> Text.toLower (Text.pack columnName) == Text.toLower x)
          $ sQLSchema_columnNames
      columnUIdToTableName =
        sQLSchema_tableNames
          `HashMap.compose` sQLSchema_columnToTable
      matchingTableNames =
        catMaybes $
          (`HashMap.lookup` columnUIdToTableName)
            <$> matchingColumnUIds
   in Text.pack tableName `elem` matchingTableNames

-- | @inSelect sqlSchema colId sel@ checks if the 'ColumnId' @colId@ is part of the 'Select' clause @sel@ in the SQLSchema @sqlSchema@.
inSelect :: ColumnId -> Select -> Bool
inSelect Star _ = True
inSelect c s =
  case s of
    Select aggs -> c `elem` go aggs
    SelectDistinct aggs -> c `elem` go aggs
  where
    go [] = []
    go (Agg _ (Column (ValColUnit ColUnit {..})) : aggs) = colUnitColId : go aggs
    go (Agg _ (Column (ValColUnit DistinctColUnit {..})) : aggs) = distinctColUnitColdId : go aggs
    go (Agg _ _ : aggs) = go aggs

data GuardTableColumnResult
  = TableNotInScope
  | ColumnNotInTable
  | ColumnInTable
  deriving stock (Eq, Show)

guardTableColumn ::
  ( Monad m,
    MonadState ParserState m
  ) =>
  SQLSchema ->
  TableId ->
  ColumnId ->
  m GuardTableColumnResult
guardTableColumn sqlSchema t c = do
  ParserState {..} <- get
  pure $
    if Left t `HashMap.member` psTables
      then
        if c `inTable'` t
          then ColumnInTable
          else ColumnNotInTable
      else TableNotInScope
  where
    inTable' = inTable sqlSchema

data GuardAliasColumnResult
  = AliasNotInScope
  | ColumnNotInAlias
  | ColumnInAlias
  deriving stock (Eq, Show)

guardAliasColumn ::
  forall m.
  ( Monad m,
    MonadState ParserState m
  ) =>
  SQLSchema ->
  Alias ->
  ColumnId ->
  m GuardAliasColumnResult
guardAliasColumn sqlSchema a c = do
  ParserState {..} <- get
  case HashMap.lookup a psAliases of
    Nothing -> pure AliasNotInScope
    Just m -> case Map.lookupLE psCurScope m of
      Nothing -> pure AliasNotInScope
      Just (_, TableUnitSQL SpiderSQL {..} _) ->
        pure $
          if c `inSelect` spiderSQLSelect
            then ColumnInAlias
            else ColumnNotInAlias
      Just (_, Table t _) ->
        pure $
          let inTable' = inTable sqlSchema
           in if c `inTable'` t
                then ColumnInAlias
                else ColumnNotInAlias

guardColumn ::
  forall m.
  ( Parsing m,
    Monad m,
    MonadState ParserState m
  ) =>
  SQLSchema ->
  ColumnId ->
  m ()
guardColumn sqlSchema = go
  where
    inTable' = inTable sqlSchema
    go Star = pure ()
    go c = do
      ParserState {..} <- get
      columnInTablesPerScope <-
        Map.fromListWith (+) . join
          <$> traverse
            ( \case
                (Left t, scopes) ->
                  pure $
                    let columnInTable = c `inTable'` t
                     in (,fromEnum columnInTable) <$> HashSet.toList scopes
                (Right s, scopes) ->
                  pure $ (,fromEnum $ c `inSelect` s) <$> HashSet.toList scopes
            )
            (HashMap.toList psTables)
      columnInAliasesPerScope <-
        Map.fromListWith (+) . join
          <$> traverse
            ( \(_a, scopes) ->
                traverse
                  ( \(scope, tu) ->
                      case tu of
                        TableUnitSQL SpiderSQL {..} _ ->
                          pure (scope, fromEnum $ c `inSelect` spiderSQLSelect)
                        Table t _ ->
                          pure $
                            let columnInTable = c `inTable'` t
                             in (scope, fromEnum columnInTable)
                  )
                  (Map.toList scopes)
            )
            (HashMap.toList psAliases)
      let columnPerScope = Map.unionWith (+) columnInTablesPerScope columnInAliasesPerScope
      unless
        ( maybe False ((== 1) . snd) $
            Map.lookupLE psCurScope columnPerScope
        )
        . unexpected
        $ "there is no single table in scope with column "
          <> show c
          <> "."

-- | @withGuards sqlSchema p@ fails conditioned on whether or not
-- all referenced columns are members of tables or aliases that are in scope.
withGuards ::
  forall m p.
  ( Parsing m,
    MonadPlus m,
    MonadState ParserState m
  ) =>
  SQLSchema ->
  m p ->
  m p
withGuards sqlSchema p = do
  pRes <- p
  ParserState {..} <- get
  let curGuards = fromMaybe mempty $ HashMap.lookup psCurScope psGuards
      f :: Guard -> m ()
      f (GuardTableColumn t c) = do
        columnInTable <- guardTableColumn sqlSchema t c
        case columnInTable of
          TableNotInScope ->
            unexpected $
              "table "
                <> show t
                <> " is not in scope."
          ColumnNotInTable ->
            unexpected $
              "column "
                <> show c
                <> " is not in table "
                <> show t
                <> "."
          ColumnInTable -> pure ()
      f (GuardAliasColumn a c) = do
        columnInAlias <- guardAliasColumn sqlSchema a c
        case columnInAlias of
          AliasNotInScope ->
            unexpected $
              "alias "
                <> show a
                <> " is not in scope."
          ColumnNotInAlias ->
            unexpected $
              "column " <> show c
                <> " is not in alias "
                <> show a
                <> "."
          ColumnInAlias -> pure ()
      f (GuardColumn c) = guardColumn sqlSchema c
  forM_ curGuards f
  pure pRes

-- | 'TableId' parser.
--
-- >>> testParseOnly tableId "singer"
-- Right (TableId {tableName = "singer"})
--
-- >>> testParseOnly tableId "Singer"
-- Right (TableId {tableName = "singer"})
--
-- >>> testParseOnly tableId "sanger"
-- Left "tableId: satisfyElem"
--
-- >>> (Atto.parseOnly . flip runReaderT (ParserEnv (ParserEnvWithGuards withGuards) sqlSchema { sQLSchema_tableNames = HashMap.union (sQLSchema_tableNames sqlSchema) (HashMap.singleton "2" "singers") }) . flip evalStateT mkParserState) (tableId <* (lift . lift) Atto.endOfInput) "singers"
-- Right (TableId {tableName = "singers"})
tableId :: forall m. (CharParsing m, MonadReader ParserEnv m, MonadPlus m) => m TableId
tableId =
  let terminate q = q <* notFollowedBy (alphaNum <|> char '_')
   in flip (<?>) "tableId" $ do
        ParserEnv _ sqlSchema <- ask
        let tableNameP' = runReaderT tableNameP sqlSchema
        TableId <$> terminate tableNameP'

-- | 'Alias' parser.
--
-- Hardcoded to start with an alphabetic character and to be at most 10 characters long.
alias :: forall m. (CharParsing m, MonadReader ParserEnv m, MonadPlus m) => m Alias
alias =
  let terminate q = q <* notFollowedBy (alphaNum <|> char '_')
      name =
        let p = satisfy ((||) <$> isAlphaNum <*> (== '_'))
         in liftA2 (:) (satisfy isAlpha) (manyAtMost (9 :: Int) p)
   in flip (<?>) "alias" $ do
        ParserEnv _ sqlSchema <- ask
        let tableNameP' = runReaderT tableNameP sqlSchema
            columnNameP' = runReaderT columnNameP sqlSchema
        Alias
          <$> ( (columnNameP' *> unexpected "alias must not be column name")
                  <|> (tableNameP' *> unexpected "alias must not be table name")
                  <|> terminate name
              )

-- | Alternative 'Alias' parser.
--
-- Hardcoded to start with a T followed by at most 9 digits.
alias' :: (CharParsing m, Monad m) => m Alias
alias' = flip (<?>) "alias" $ do
  _ <- satisfy (\c -> toLower c == 't')
  let terminate q = q <* notFollowedBy (alphaNum <|> char '_')
  digits <- terminate $ liftA2 (:) digit (manyAtMost (9 :: Int) digit)
  pure . Alias $ "T" <> digits

-- | 'ColumnId' parser.
--
-- >>> testParseOnly columnId "*"
-- Right Star
--
-- >>> testParseOnly columnId "invalid_column"
-- Left "columnId: satisfyElem"
--
-- >>> testParseOnly columnId "Birth_Year"
-- Right (ColumnId {columnName = "Birth_Year"})
--
-- >>> testParseOnly columnId "birth_year"
-- Right (ColumnId {columnName = "Birth_Year"})
columnId :: forall m. (CharParsing m, MonadReader ParserEnv m, MonadPlus m) => m ColumnId
columnId =
  let terminate q = q <* notFollowedBy (alphaNum <|> char '_')
   in flip (<?>) "columnId" $ do
        ParserEnv _ sqlSchema <- ask
        let columnNameP' = runReaderT columnNameP sqlSchema
        isStar $> Star <|> (ColumnId <$> terminate columnNameP')

-- | 'From' parser.
--
-- >>> testParseOnly from "FROM singer"
-- Right (From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing})
--
-- >>> testParseOnly from "FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID"
-- Right (From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))})
from ::
  forall m.
  ( TokenParsing m,
    MonadSQL m
  ) =>
  m From
from = flip (<?>) "from" $ do
  _ <- isFrom
  someSpace
  uncurry mkFrom <$> p
  where
    p :: m (TableUnit, [(TableUnit, Maybe Cond)])
    p =
      (,)
        <$> tableUnit
        <*> many
          ( someSpace
              *> isJoin
              *> someSpace
              *> ( (,)
                     <$> tableUnit
                     <*> optional
                       ( someSpace
                           *> isOn
                           *> someSpace
                           *> cond
                       )
                 )
          )
    mkFrom :: TableUnit -> [(TableUnit, Maybe Cond)] -> From
    mkFrom tu tus =
      From
        (tu : fmap fst tus)
        ( foldl'
            ( \a b ->
                case (a, b) of
                  (Just c, Just c') -> Just (And c c')
                  (Just c, Nothing) -> Just c
                  (Nothing, Just c') -> Just c'
                  (Nothing, Nothing) -> Nothing
            )
            Nothing
            (fmap snd tus)
        )

updateAliases ::
  forall m.
  ( Parsing m,
    MonadSQL m
  ) =>
  Alias ->
  TableUnit ->
  m ()
updateAliases a tu = do
  ParserState {..} <- get
  hasConflict <-
    maybe False (Map.member psCurScope)
      . HashMap.lookup a
      . (^. field @"psAliases")
      <$> get
  when hasConflict
    . unexpected
    $ "the alias "
      <> show a
      <> "is already in this scope."
  let v = Map.singleton psCurScope tu
  modify (field @"psAliases" %~ HashMap.insertWith Map.union a v)
  let curGuards = fromMaybe mempty $ HashMap.lookup psCurScope psGuards
      f (GuardAliasColumn a' c) | a == a' = do
        ParserEnv _ sqlSchema <- ask
        columnInAlias <- guardAliasColumn sqlSchema a c
        case columnInAlias of
          AliasNotInScope -> error "impossible"
          ColumnNotInAlias ->
            unexpected $
              "column "
                <> show c
                <> " is not in alias "
                <> show a
                <> "."
          ColumnInAlias -> pure ()
      f _ = pure ()
  forM_ curGuards f

updateTables ::
  forall m.
  ( Parsing m,
    MonadSQL m
  ) =>
  Either TableId Select ->
  m ()
updateTables (Left t) = do
  ParserState {..} <- get
  hasConflict <-
    maybe False (HashSet.member psCurScope)
      . HashMap.lookup (Left t)
      . (^. field @"psTables")
      <$> get
  when hasConflict
    . unexpected
    $ "the table "
      <> show t
      <> "is already in this scope."
  let v = HashSet.singleton psCurScope
  modify (field @"psTables" %~ HashMap.insertWith HashSet.union (Left t) v)
  let curGuards = fromMaybe mempty $ HashMap.lookup psCurScope psGuards
      f (GuardTableColumn t' c) | t == t' = do
        ParserEnv _ sqlSchema <- ask
        columnInTable <- guardTableColumn sqlSchema t c
        case columnInTable of
          TableNotInScope -> error "impossible"
          ColumnNotInTable ->
            unexpected $
              "column "
                <> show c
                <> " is not in table "
                <> show t
                <> "."
          ColumnInTable -> pure ()
      f _ = pure ()
  forM_ curGuards f
updateTables (Right s) = do
  ParserState {..} <- get
  let v = HashSet.singleton psCurScope
  modify (field @"psTables" %~ HashMap.insertWith HashSet.union (Right s) v)

-- | 'TableUnit' parser.
--
-- >>> testParseOnly tableUnit "song as t1"
-- Right (Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T1"})))
--
-- >>> testParseOnly tableUnit "(SELECT * FROM song)"
-- Right (TableUnitSQL (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}) Nothing)
--
-- >>> testParseOnly tableUnit "(SELECT * FROM song) as t1"
-- Right (TableUnitSQL (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}) (Just (Alias {aliasName = "T1"})))
tableUnit ::
  forall m.
  ( TokenParsing m,
    MonadSQL m
  ) =>
  m TableUnit
tableUnit =
  flip (<?>) "tableUnit" $
    let aliasP = someSpace *> isAs *> someSpace *> alias'
        tableUnitSQL =
          flip (<?>) "tableUnitSQL" $
            TableUnitSQL
              <$> betweenParentheses (get >>= spiderSQL . (field @"psCurScope" %~ succ))
                <*> optional aliasP
        table =
          flip (<?>) "table" $
            Table
              <$> tableId
                <*> optional aliasP
     in do
          tu <- tableUnitSQL <|> table
          case tu of
            TableUnitSQL _ (Just a) -> updateAliases a tu
            TableUnitSQL SpiderSQL {..} Nothing -> updateTables (Right spiderSQLSelect)
            Table _ (Just a) -> updateAliases a tu
            Table t Nothing -> updateTables (Left t)
          pure tu

-- | 'Cond' parser.
--
-- >>> testParseOnly cond "t1.Singer_ID = t2.Singer_ID"
-- Right (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly cond "t1.Singer_ID + t2.Singer_ID = t2.Singer_ID"
-- Right (Eq (Plus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly cond "t1.Singer_ID = t2.Singer_ID + t2.Singer_ID"
-- Right (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Plus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly cond "t2.Name = \"Adele\" AND t3.Name = \"Beyoncé\""
-- Right (And (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Name"}}})) (Column (ValString {stringValue = "Adele"}))) (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T3"})), colUnitColId = ColumnId {columnName = "Name"}}})) (Column (ValString {stringValue = "Beyonc\233"}))))
--
-- >>> testParseOnly cond "song_id IN (SELECT song_id FROM song)"
-- Right (In (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}})) (Column (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly cond "song_id NOT IN (SELECT song_id FROM song)"
-- Right (Not (In (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}})) (Column (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}}))))
--
-- >>> testParseOnly cond "t1.Singer_ID - t2.Singer_ID = (select song_id - song_id from song order by song_id - song_id desc)"
-- Right (Eq (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly cond "t1.Singer_ID - t2.Singer_ID = (select song_id - song_id from song order by (song_id - song_id) desc)"
-- Right (Eq (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly cond "t1.Singer_ID - t2.Singer_ID = (select (song_id - song_id) from song order by (song_id - song_id) desc)"
-- Right (Eq (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
--
-- >>> testParseOnly cond "(t1.Singer_ID - t2.Singer_ID) = (select (song_id - song_id) from song order by (song_id - song_id) desc)"
-- Right (Eq (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Minus (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}) (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})))
cond :: (TokenParsing m, MonadSQL m) => m Cond
cond =
  flip (<?>) "cond" $
    let mkCond p' =
          let suffix r' =
                let q = mkCond p'
                 in choice
                      [ And r' <$> (someSpace *> isAnd *> someSpace *> q),
                        Or r' <$> (someSpace *> isOr *> someSpace *> q)
                      ]
              suffixRec base = do
                c <- base
                suffixRec (suffix c) <|> pure c
              r =
                choice
                  [ Not <$> (isNot *> spaces *> p'),
                    p'
                  ]
           in suffixRec r
        p =
          choice
            [ binary Eq isEq,
              binary Ge isGe,
              binary Le isLe,
              binary Gt isGt,
              binary Lt isLt,
              binary Ne isNe,
              binary In isIn,
              binaryNot In isIn,
              binary Like isLike,
              binaryNot Like isLike,
              Between
                <$> betweenOptionalParentheses valUnit
                <*> (someSpace *> isBetween *> someSpace *> betweenOptionalParentheses valUnit)
                <*> (someSpace *> isAnd *> someSpace *> betweenOptionalParentheses valUnit)
            ]
        binary f q =
          f <$> betweenOptionalParentheses valUnit
            <*> (spaces *> q *> spaces *> betweenOptionalParentheses valUnit)
        binaryNot f q =
          (Not .) . f <$> betweenOptionalParentheses valUnit
            <*> (spaces *> isNot *> someSpace *> q *> spaces *> betweenOptionalParentheses valUnit)
     in mkCond p

-- | 'Val' parser.
--
-- >>> testParseOnly val "count song.Song_ID"
-- Right (ValColUnit {columnValue = ColUnit {colUnitAggId = Just Count, colUnitTable = Just (Left (TableId {tableName = "song"})), colUnitColId = ColumnId {columnName = "Song_ID"}}})
--
-- >>> testParseOnly val "count(song.Song_ID)"
-- Right (ValColUnit {columnValue = ColUnit {colUnitAggId = Just Count, colUnitTable = Just (Left (TableId {tableName = "song"})), colUnitColId = ColumnId {columnName = "Song_ID"}}})
--
-- >>> testParseOnly val "(select *)"
-- Right (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})
--
-- >>> testParseOnly val "(select song_id from song)"
-- Right (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}})
val ::
  forall m.
  ( TokenParsing m,
    MonadSQL m
  ) =>
  m Val
val = flip (<?>) "val" $ choice choices
  where
    terminate q = q <* notFollowedBy (alphaNum <|> char '_')
    choices = [valColUnit, number, valString, valSQL]
    valColUnit = ValColUnit <$> colUnit
    number = Number <$> terminate (doubleP 16)
    valString = ValString <$> terminate (quotedString 64)
    valSQL =
      ValSQL
        <$> betweenParentheses
          (get >>= spiderSQL . (field @"psCurScope" %~ succ))

-- | Parser for WHERE clauses.
--
-- >>> testParseOnly whereCond "where t1.Singer_ID = t2.Singer_ID"
-- Right (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))
--
-- >>> testParseOnly whereCond "where Singer_ID = 1"
-- Right (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (Number {numberValue = 1.0})))
whereCond :: (TokenParsing m, MonadSQL m) => m Cond
whereCond = flip (<?>) "whereCond" $ isWhere *> someSpace *> cond

-- | Parser for group-by clauses.
--
-- >>> testParseOnly groupBy "group by t1.Song_ID"
-- Right [ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Song_ID"}}]
--
-- >>> testParseOnly groupBy "group by t1.Song_ID, t2.Singer_ID"
-- Right [ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Song_ID"}},ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}]
--
-- >>> testParseOnly groupBy "group by count t1.Song_ID, t2.Singer_ID"
-- Right [ColUnit {colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Song_ID"}},ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}]
groupBy ::
  forall m.
  ( TokenParsing m,
    MonadSQL m
  ) =>
  m [ColUnit]
groupBy =
  flip (<?>) "groupBy" $
    isGroupBy
      *> someSpace
      *> sepBy1 (betweenOptionalParentheses colUnit) (spaces *> isComma <* someSpace)

-- | 'OrderBy' Parser.
--
-- >>> testParseOnly orderBy "order by t1.Song_ID, t2.Singer_ID desc"
-- Right (OrderBy [(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Song_ID"}}}),Asc),(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}),Desc)])
--
-- >>> testParseOnly orderBy "order by t1.Song_ID asc, t2.Singer_ID desc"
-- Right (OrderBy [(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Song_ID"}}}),Asc),(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}}),Desc)])
--
-- >>> testParseOnly orderBy "order by count(t1.Song_ID) desc"
-- Right (OrderBy [(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Song_ID"}}}),Desc)])
--
-- >>> testParseOnly orderBy "order by sum(t1.Song_ID)"
-- Right (OrderBy [(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Just Sum, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Song_ID"}}}),Asc)])
orderBy :: forall m. (TokenParsing m, MonadSQL m) => m OrderBy
orderBy = flip (<?>) "orderBy" $ do
  _ <- isOrderBy
  someSpace
  valUnits <-
    let order = optional (spaces *> (isAsc $> Asc <|> isDesc $> Desc)) >>= maybe (pure Asc) pure
        p = (,) <$> betweenOptionalParentheses valUnit <*> order
     in sepBy1 p (spaces *> isComma <* someSpace)
  pure $ OrderBy valUnits

-- | Parser for HAVING clauses.
--
-- >>> testParseOnly havingCond "having count(t1.Sales) = 10"
-- Right (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Sales"}}})) (Column (Number {numberValue = 10.0})))
havingCond :: forall m. (TokenParsing m, MonadSQL m) => m Cond
havingCond = flip (<?>) "havingCond" $ isHaving *> someSpace *> cond

-- | Parser for LIMIT clauses.
--
-- >>> testParseOnly limit "limit 10"
-- Right 10
limit :: forall m. (TokenParsing m, Monad m) => m Int
limit = flip (<?>) "limit" $ isLimit *> someSpace *> intP 8

-- | 'SpiderSQL' parser.
--
-- >>> spiderSQLTestParseOnly "select *"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select count(*)"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select * from singer"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select * from song"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT T1.Name, T1.Citizenship, T1.Birth_Year FROM singer AS T1 ORDER BY T1.Birth_Year DESC"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Name"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Citizenship"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Birth_Year"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"}))], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Birth_Year"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT Name, Citizenship, Birth_Year FROM singer ORDER BY Birth_Year DESC"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Name"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Citizenship"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Birth_Year"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Birth_Year"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT   name  ,    citizenship  ,   birth_year   FROM   Singer  ORDER BY   birth_year   DESC"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Name"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Citizenship"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Birth_Year"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Just (OrderBy [(Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Birth_Year"}}}),Desc)]), spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT T2.Title, T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Title"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Name"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT T1.Name FROM singer AS T1 JOIN song AS T2 ON T1.Singer_ID = T2.Singer_ID GROUP BY T1.Name HAVING COUNT(*) > 1"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Name"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Name"}}], spiderSQLOrderBy = Nothing, spiderSQLHaving = Just (Gt (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Just Count, colUnitTable = Nothing, colUnitColId = Star}})) (Column (Number {numberValue = 1.0}))), spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "select count t2.Song_ID, t1.Citizenship from singer AS t1 JOIN song AS t2 on t1.Singer_ID = t2.Singer_ID group by count t2.Song_ID, t1.Citizenship"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Song_ID"}}})),Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Citizenship"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "singer"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Just (Eq (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Singer_ID"}}})))}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [ColUnit {colUnitAggId = Just Count, colUnitTable = Just (Right (Alias {aliasName = "T2"})), colUnitColId = ColumnId {columnName = "Song_ID"}},ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Citizenship"}}], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParseOnly "SELECT title FROM song WHERE song_id IN (SELECT song_id FROM song)"
-- Right (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Title"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Just (In (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}})) (Column (ValSQL {sqlValue = SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = ColumnId {columnName = "Song_ID"}}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) Nothing], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing}}))), spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
--
-- >>> spiderSQLTestParse "SELECT T1.title ,  count(*) FROM song AS T1 JOIN song AS T2 ON T1.song id"
-- Done " ON T1.song id" (SpiderSQL {spiderSQLSelect = Select [Agg Nothing (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Just (Right (Alias {aliasName = "T1"})), colUnitColId = ColumnId {columnName = "Title"}}})),Agg (Just Count) (Column (ValColUnit {columnValue = ColUnit {colUnitAggId = Nothing, colUnitTable = Nothing, colUnitColId = Star}}))], spiderSQLFrom = From {fromTableUnits = [Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T1"})),Table (TableId {tableName = "song"}) (Just (Alias {aliasName = "T2"}))], fromCond = Nothing}, spiderSQLWhere = Nothing, spiderSQLGroupBy = [], spiderSQLOrderBy = Nothing, spiderSQLHaving = Nothing, spiderSQLLimit = Nothing, spiderSQLIntersect = Nothing, spiderSQLExcept = Nothing, spiderSQLUnion = Nothing})
spiderSQL ::
  forall m.
  ( TokenParsing m,
    MonadPlus m,
    MonadReader ParserEnv m
  ) =>
  ParserState ->
  m SpiderSQL
spiderSQL env =
  flip (<?>) "spiderSQL" $
    flip evalStateT env $ do
      ParserEnv (ParserEnvWithGuards peWithGuards) sqlSchema <- ask
      sel <- select
      fro <- peWithGuards sqlSchema $ fromMaybe (From [] Nothing) <$> optional (spaces *> from)
      whe <- optional (peWithGuards sqlSchema $ someSpace *> whereCond)
      grp <- fromMaybe [] <$> optional (peWithGuards sqlSchema $ someSpace *> groupBy)
      (ord, hav) <-
        permute
          ( (,) <$$> optional (peWithGuards sqlSchema $ someSpace *> orderBy)
              <||> optional (peWithGuards sqlSchema $ someSpace *> havingCond)
          )
      lim <- optional (someSpace *> limit)
      (int, exc, uni) <-
        permute
          ( (,,) <$$> optional (someSpace *> isIntersect *> someSpace *> spiderSQL env)
              <||> optional (someSpace *> isExcept *> someSpace *> spiderSQL env)
              <||> optional (someSpace *> isUnion *> someSpace *> spiderSQL env)
          )
      pure $ SpiderSQL sel fro whe grp ord hav lim int exc uni
