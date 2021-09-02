{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.StormRecord where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

stormRecordSchema :: SQLSchema
stormRecordSchema =
  let columnNames = HashMap.fromList [("7", "Region_id"), ("12", "Number_city_affected"), ("1", "Storm_ID"), ("4", "Max_speed"), ("2", "Name"), ("5", "Damage_millions_USD"), ("8", "Region_code"), ("11", "Storm_ID"), ("3", "Dates_active"), ("6", "Number_Deaths"), ("9", "Region_name"), ("10", "Region_id")]
      tableNames = HashMap.fromList [("0", "storm"), ("1", "region"), ("2", "affected_region")]
      columnToTable = HashMap.fromList [("7", "1"), ("12", "2"), ("1", "0"), ("4", "0"), ("2", "0"), ("5", "0"), ("8", "1"), ("11", "2"), ("3", "0"), ("6", "0"), ("9", "1"), ("10", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6"]), ("1", ["7", "8", "9"]), ("2", ["10", "11", "12"])]
      foreignKeys = HashMap.fromList [("11", "1"), ("10", "7")]
      foreignKeysTables = HashMap.fromList [("2", ["0", "1"])]
      primaryKeys = ["1", "7", "10"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

stormRecordQueries :: [Text.Text]
stormRecordQueries =
  [ "select avg(damage_millions_usd), max(damage_millions_usd) from storm where max_speed > 1000",
    "select sum(number_deaths), sum(damage_millions_usd) from storm where max_speed > (select avg(max_speed) from storm)",
    "select name, damage_millions_usd from storm order by max_speed desc"
  ]

stormRecordQueriesFails :: [Text.Text]
stormRecordQueriesFails = []

stormRecordParserTests :: TestItem
stormRecordParserTests =
  Group "stormRecord" $
    (ParseQueryExprWithGuards stormRecordSchema <$> stormRecordQueries)
      <> (ParseQueryExprWithoutGuards stormRecordSchema <$> stormRecordQueries)
      <> (ParseQueryExprFails stormRecordSchema <$> stormRecordQueriesFails)

stormRecordLexerTests :: TestItem
stormRecordLexerTests =
  Group "stormRecord" $
    LexQueryExpr stormRecordSchema <$> stormRecordQueries
