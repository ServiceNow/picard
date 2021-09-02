{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Geo where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

geoSchema :: SQLSchema
geoSchema =
  let columnNames = HashMap.fromList [("15", "lowest_point"), ("7", "city_name"), ("25", "state_name"), ("28", "country_name"), ("13", "state_name"), ("14", "highest_elevation"), ("22", "mountain_name"), ("19", "area"), ("29", "traverse"), ("12", "border"), ("17", "lowest_elevation"), ("1", "state_name"), ("23", "mountain_altitude"), ("18", "lake_name"), ("4", "country_name"), ("26", "river_name"), ("16", "highest_point"), ("2", "population"), ("20", "country_name"), ("5", "capital"), ("27", "length"), ("8", "population"), ("11", "state_name"), ("3", "area"), ("21", "state_name"), ("24", "country_name"), ("6", "density"), ("9", "country_name"), ("10", "state_name")]
      tableNames = HashMap.fromList [("0", "state"), ("1", "city"), ("4", "lake"), ("2", "border_info"), ("5", "mountain"), ("3", "highlow"), ("6", "river")]
      columnToTable = HashMap.fromList [("15", "3"), ("7", "1"), ("25", "5"), ("28", "6"), ("13", "3"), ("14", "3"), ("22", "5"), ("19", "4"), ("29", "6"), ("12", "2"), ("17", "3"), ("1", "0"), ("23", "5"), ("18", "4"), ("4", "0"), ("26", "6"), ("16", "3"), ("2", "0"), ("20", "4"), ("5", "0"), ("27", "6"), ("8", "1"), ("11", "2"), ("3", "0"), ("21", "4"), ("24", "5"), ("6", "0"), ("9", "1"), ("10", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6"]), ("1", ["7", "8", "9", "10"]), ("4", ["18", "19", "20", "21"]), ("2", ["11", "12"]), ("5", ["22", "23", "24", "25"]), ("3", ["13", "14", "15", "16", "17"]), ("6", ["26", "27", "28", "29"])]
      foreignKeys = HashMap.fromList [("25", "1"), ("13", "1"), ("29", "1"), ("12", "1"), ("11", "1"), ("10", "1")]
      foreignKeysTables = HashMap.fromList [("1", ["0"]), ("2", ["0"]), ("5", ["0"]), ("3", ["0"]), ("6", ["0"])]
      primaryKeys = ["1", "7", "12", "13", "22", "26"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

geoQueries :: [Text.Text]
geoQueries =
  [ "select river_name from river group by river_name order by count ( distinct traverse ) desc limit 1;",
    "select river_name from river group by ( river_name ) order by count ( distinct traverse ) desc limit 1;",
    "select t1.capital from highlow as t2 join state as t1 on t1.state_name = t2.state_name where t2.lowest_elevation = ( select min ( lowest_elevation ) from highlow );",
    "select t1.capital from highlow as t2 join state as t1 on t1.state_name = t2.state_name where t2.lowest_elevation = ( select min ( lowest_elevation ) from highlow ) ;"
  ]

geoQueriesFails :: [Text.Text]
geoQueriesFails = []

geoParserTests :: TestItem
geoParserTests =
  Group "geo" $
    (ParseQueryExprWithGuards geoSchema <$> geoQueries)
      <> (ParseQueryExprWithoutGuards geoSchema <$> geoQueries)
      <> (ParseQueryExprFails geoSchema <$> geoQueriesFails)

geoLexerTests :: TestItem
geoLexerTests =
  Group "geo" $
    LexQueryExpr geoSchema <$> geoQueries
