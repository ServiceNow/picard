{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Bike1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

bike1Schema :: SQLSchema
bike1Schema =
  let columnNames = HashMap.fromList [("15", "start_station_name"), ("37", "mean_visibility_miles"), ("7", "installation_date"), ("25", "mean_temperature_f"), ("43", "cloud_cover"), ("28", "mean_dew_point_f"), ("13", "duration"), ("31", "mean_humidity"), ("14", "start_date"), ("36", "max_visibility_miles"), ("22", "zip_code"), ("19", "end_station_id"), ("44", "events"), ("29", "min_dew_point_f"), ("12", "id"), ("30", "max_humidity"), ("17", "end_date"), ("35", "min_sea_level_pressure_inches"), ("45", "wind_dir_degrees"), ("1", "id"), ("23", "date"), ("18", "end_station_name"), ("40", "mean_wind_speed_mph"), ("4", "long"), ("26", "min_temperature_f"), ("16", "start_station_id"), ("34", "mean_sea_level_pressure_inches"), ("2", "name"), ("20", "bike_id"), ("39", "max_wind_Speed_mph"), ("46", "zip_code"), ("5", "dock_count"), ("27", "max_dew_point_f"), ("41", "max_gust_speed_mph"), ("8", "station_id"), ("11", "time"), ("33", "max_sea_level_pressure_inches"), ("38", "min_visibility_miles"), ("3", "lat"), ("21", "subscription_type"), ("24", "max_temperature_f"), ("42", "precipitation_inches"), ("6", "city"), ("9", "bikes_available"), ("10", "docks_available"), ("32", "min_humidity")]
      tableNames = HashMap.fromList [("0", "station"), ("1", "status"), ("2", "trip"), ("3", "weather")]
      columnToTable = HashMap.fromList [("15", "2"), ("37", "3"), ("7", "0"), ("25", "3"), ("43", "3"), ("28", "3"), ("13", "2"), ("31", "3"), ("14", "2"), ("36", "3"), ("22", "2"), ("19", "2"), ("44", "3"), ("29", "3"), ("12", "2"), ("30", "3"), ("17", "2"), ("35", "3"), ("45", "3"), ("1", "0"), ("23", "3"), ("18", "2"), ("40", "3"), ("4", "0"), ("26", "3"), ("16", "2"), ("34", "3"), ("2", "0"), ("20", "2"), ("39", "3"), ("46", "3"), ("5", "0"), ("27", "3"), ("41", "3"), ("8", "1"), ("11", "1"), ("33", "3"), ("38", "3"), ("3", "0"), ("21", "2"), ("24", "3"), ("42", "3"), ("6", "0"), ("9", "1"), ("10", "1"), ("32", "3")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7"]), ("1", ["8", "9", "10", "11"]), ("2", ["12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]), ("3", ["23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46"])]
      foreignKeys = HashMap.fromList [("8", "1")]
      foreignKeysTables = HashMap.fromList [("1", ["0"])]
      primaryKeys = ["1", "12"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

bike1Queries :: [Text.Text]
bike1Queries =
  [ "select t1.date from weather as t1 where t1.max_temperature_f > 85",
    "select weather.date from weather where weather.max_temperature_f > 85",
    "select date from weather where weather.max_temperature_f > 85",
    "select weather.date from weather where max_temperature_f > 85",
    "select date from weather where precipitation_inches > 85",
    "select date from weather where max_temperature_f > 85",
    "select date, zip_code from weather where max_temperature_f >= 80",
    "select zip_code, count(*) from weather where max_wind_speed_mph >= 25 group by zip_code",
    "select date, zip_code from weather where min_dew_point_f < (select min(min_dew_point_f) from weather where zip_code = 94107)",
    "select date, mean_temperature_f, mean_humidity from weather order by max_gust_speed_mph desc limit 3",
    "select distinct zip_code from weather except select distinct zip_code from weather where max_dew_point_f >= 70",
    "select date, max_temperature_f - min_temperature_f from weather order by max_temperature_f - min_temperature_f limit 1"
  ]

bike1QueriesFails :: [Text.Text]
bike1QueriesFails = []

bike1ParserTests :: TestItem
bike1ParserTests =
  Group "bike1" $
    (ParseQueryExprWithGuards bike1Schema <$> bike1Queries)
      <> (ParseQueryExprWithoutGuards bike1Schema <$> bike1Queries)
      <> (ParseQueryExprFails bike1Schema <$> bike1QueriesFails)

bike1LexerTests :: TestItem
bike1LexerTests =
  Group "bike1" $
    LexQueryExpr bike1Schema <$> bike1Queries
