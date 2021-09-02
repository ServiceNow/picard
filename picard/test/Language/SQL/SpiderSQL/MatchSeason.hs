{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.MatchSeason where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

matchSeasonSchema :: SQLSchema
matchSeasonSchema =
  let columnNames = HashMap.fromList [("15", "Player_ID"), ("7", "Season"), ("13", "Draft_Class"), ("14", "College"), ("19", "Singles_WL"), ("12", "Draft_Pick_Number"), ("17", "Years_Played"), ("1", "Country_id"), ("18", "Total_WL"), ("4", "Official_native_language"), ("16", "Player"), ("2", "Country_name"), ("20", "Doubles_WL"), ("5", "Team_id"), ("8", "Player"), ("11", "Team"), ("3", "Capital"), ("21", "Team"), ("6", "Name"), ("9", "Position"), ("10", "Country")]
      tableNames = HashMap.fromList [("0", "country"), ("1", "team"), ("2", "match_season"), ("3", "player")]
      columnToTable = HashMap.fromList [("15", "3"), ("7", "2"), ("13", "2"), ("14", "2"), ("19", "3"), ("12", "2"), ("17", "3"), ("1", "0"), ("18", "3"), ("4", "0"), ("16", "3"), ("2", "0"), ("20", "3"), ("5", "1"), ("8", "2"), ("11", "2"), ("3", "0"), ("21", "3"), ("6", "1"), ("9", "2"), ("10", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4"]), ("1", ["5", "6"]), ("2", ["7", "8", "9", "10", "11", "12", "13", "14"]), ("3", ["15", "16", "17", "18", "19", "20", "21"])]
      foreignKeys = HashMap.fromList [("11", "5"), ("21", "5"), ("10", "1")]
      foreignKeysTables = HashMap.fromList [("2", ["0", "1"]), ("3", ["1"])]
      primaryKeys = ["1", "5", "7", "15"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

matchSeasonQueries :: [Text.Text]
matchSeasonQueries =
  [ "select college from match_season group by college having count(*) >= 2 order by college desc"
  ]

matchSeasonQueriesFails :: [Text.Text]
matchSeasonQueriesFails = []

matchSeasonParserTests :: TestItem
matchSeasonParserTests =
  Group "matchSeason" $
    (ParseQueryExprWithGuards matchSeasonSchema <$> matchSeasonQueries)
      <> (ParseQueryExprWithoutGuards matchSeasonSchema <$> matchSeasonQueries)
      <> (ParseQueryExprFails matchSeasonSchema <$> matchSeasonQueriesFails)

matchSeasonLexerTests :: TestItem
matchSeasonLexerTests =
  Group "matchSeason" $
    LexQueryExpr matchSeasonSchema <$> matchSeasonQueries
