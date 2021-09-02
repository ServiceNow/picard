module Main where

import Control.Applicative (optional)
import Control.Monad.Reader (runReaderT)
import Control.Monad.Trans (lift)
import qualified Data.Attoparsec.Text as Atto (char, endOfInput, parseOnly, skipSpace)
import Language.SQL.SpiderSQL.Academic (academicLexerTests, academicParserTests)
import Language.SQL.SpiderSQL.AssetsMaintenance (assetsMaintenanceLexerTests, assetsMaintenanceParserTests)
import Language.SQL.SpiderSQL.Bike1 (bike1LexerTests, bike1ParserTests)
import Language.SQL.SpiderSQL.Car1 (car1LexerTests, car1ParserTests)
import Language.SQL.SpiderSQL.Chinook1 (chinook1LexerTests, chinook1ParserTests)
import Language.SQL.SpiderSQL.ConcertSinger (concertSingerLexerTests, concertSingerParserTests)
import Language.SQL.SpiderSQL.DepartmentManagement (departmentManagementLexerTests, departmentManagementParserTests)
import Language.SQL.SpiderSQL.Flight1 (flight1LexerTests, flight1ParserTests)
import Language.SQL.SpiderSQL.Geo (geoLexerTests, geoParserTests)
import Language.SQL.SpiderSQL.Inn1 (inn1LexerTests, inn1ParserTests)
import Language.SQL.SpiderSQL.Lexer (lexSpiderSQL)
import Language.SQL.SpiderSQL.MatchSeason (matchSeasonLexerTests, matchSeasonParserTests)
import Language.SQL.SpiderSQL.Parse (ParserEnv (..), ParserEnvWithGuards (..), mkParserState, spiderSQL, withGuards)
import Language.SQL.SpiderSQL.Pets1 (pets1LexerTests, pets1ParserTests)
import Language.SQL.SpiderSQL.PhoneMarket (phoneMarketLexerTests, phoneMarketParserTests)
import Language.SQL.SpiderSQL.ProductCatalog (productCatalogLexerTests, productCatalogParserTests)
import Language.SQL.SpiderSQL.Scholar (scholarLexerTests, scholarParserTests)
import Language.SQL.SpiderSQL.StormRecord (stormRecordLexerTests, stormRecordParserTests)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import qualified Test.Tasty as T
import qualified Test.Tasty.HUnit as H

-- | Run 'cabal repl test:picard-test' to get a REPL for the tests.
main :: IO ()
main = T.defaultMain testTree

testData :: TestItem
testData =
  Group
    "tests"
    [ Group
        "lexer"
        [ academicLexerTests,
          assetsMaintenanceLexerTests,
          bike1LexerTests,
          car1LexerTests,
          chinook1LexerTests,
          concertSingerLexerTests,
          departmentManagementLexerTests,
          flight1LexerTests,
          geoLexerTests,
          inn1LexerTests,
          matchSeasonLexerTests,
          pets1LexerTests,
          phoneMarketLexerTests,
          productCatalogLexerTests,
          scholarLexerTests,
          stormRecordLexerTests
        ],
      Group
        "parser"
        [ academicParserTests,
          assetsMaintenanceParserTests,
          bike1ParserTests,
          car1ParserTests,
          chinook1ParserTests,
          concertSingerParserTests,
          departmentManagementParserTests,
          flight1ParserTests,
          geoParserTests,
          inn1ParserTests,
          matchSeasonParserTests,
          pets1ParserTests,
          phoneMarketParserTests,
          productCatalogParserTests,
          scholarParserTests,
          stormRecordParserTests
        ]
    ]

testTree :: T.TestTree
testTree = toTest testData
  where
    parseOnly p parserEnv query =
      Atto.parseOnly
        ( runReaderT
            ( p
                <* optional (lift $ Atto.skipSpace <* Atto.char ';')
                <* lift Atto.endOfInput
            )
            parserEnv
        )
        query
    toTest (Group name tests) =
      T.testGroup name $ toTest <$> tests
    toTest (LexQueryExpr sqlSchema query) =
      H.testCase ("Lex " <> show query) $
        case parseOnly lexSpiderSQL sqlSchema query of
          Left e -> H.assertFailure e
          Right _ -> pure ()
    toTest (ParseQueryExprWithoutGuards sqlSchema query) =
      H.testCase ("Parse " <> show query) $
        case parseOnly
          (spiderSQL mkParserState)
          (ParserEnv (ParserEnvWithGuards (const id)) sqlSchema)
          query of
          Left e -> H.assertFailure e
          Right _ -> pure ()
    toTest (ParseQueryExprWithGuards sqlSchema query) =
      H.testCase ("Parse " <> show query) $
        case parseOnly
          (spiderSQL mkParserState)
          (ParserEnv (ParserEnvWithGuards withGuards) sqlSchema)
          query of
          Left e -> H.assertFailure e
          Right _ -> pure ()
    toTest (ParseQueryExprFails sqlSchema query) =
      H.testCase ("Fail " <> show query) $
        case parseOnly
          (spiderSQL mkParserState)
          (ParserEnv (ParserEnvWithGuards withGuards) sqlSchema)
          query of
          Left _ -> pure ()
          Right a -> H.assertFailure $ show a
