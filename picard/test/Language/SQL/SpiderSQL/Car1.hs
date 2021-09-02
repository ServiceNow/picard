{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Car1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

car1Schema :: SQLSchema
car1Schema =
  let columnNames = HashMap.fromList [("1", "ContId"), ("10", "ModelId"), ("11", "Maker"), ("12", "Model"), ("13", "MakeId"), ("14", "Model"), ("15", "Make"), ("16", "Id"), ("17", "MPG"), ("18", "Cylinders"), ("19", "Edispl"), ("2", "Continent"), ("20", "Horsepower"), ("21", "Weight"), ("22", "Accelerate"), ("23", "Year"), ("3", "CountryId"), ("4", "CountryName"), ("5", "Continent"), ("6", "Id"), ("7", "Maker"), ("8", "FullName"), ("9", "Country")]
      tableNames = HashMap.fromList [("0", "continents"), ("1", "countries"), ("2", "car_makers"), ("3", "model_list"), ("4", "car_names"), ("5", "cars_data")]
      columnToTable = HashMap.fromList [("1", "0"), ("10", "3"), ("11", "3"), ("12", "3"), ("13", "4"), ("14", "4"), ("15", "4"), ("16", "5"), ("17", "5"), ("18", "5"), ("19", "5"), ("2", "0"), ("20", "5"), ("21", "5"), ("22", "5"), ("23", "5"), ("3", "1"), ("4", "1"), ("5", "1"), ("6", "2"), ("7", "2"), ("8", "2"), ("9", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2"]), ("1", ["3", "4", "5"]), ("2", ["6", "7", "8", "9"]), ("3", ["10", "11", "12"]), ("4", ["13", "14", "15"]), ("5", ["16", "17", "18", "19", "20", "21", "22", "23"])]
      foreignKeys = mempty
      foreignKeysTables = mempty
      primaryKeys = mempty
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

car1Queries :: [Text.Text]
car1Queries =
  [ "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryid having count(*) > 3",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryid having count(*) > 3 union select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country join model_list as t3 on t2.id = t3.maker where t3.model = \"Fiat\"",
    "select modelid from model_list",
    "select modelid from model_list where modelid in (select modelid from model_list)",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list)))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list))))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list)))))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list))))))",
    "select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list where modelid in (select modelid from model_list)))))))"
  ]

car1QueriesFails :: [Text.Text]
car1QueriesFails =
  [ "select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.country",
    "select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.country where t2.make = \"Fiat\"",
    "select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryid having count(*) > 3 union select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.country where t2.make = \"Fiat\"",
    "select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data where horsepower = (select max(horsepower) from cars_data where year = (select max(horsepower) from cars_data where accelerate = (select max(accelerate) from cars_data where horses",
    "select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data where horsepower = (select max(horsepower) from cars_data where model = (select id from cars_data where model = (select id from cars_data where model = (select id from cars_data where model = (select id from cars_data where horsepower = (select max(h"
  ]

car1ParserTests :: TestItem
car1ParserTests =
  Group "car1" $
    (ParseQueryExprWithGuards car1Schema <$> car1Queries)
      <> (ParseQueryExprWithoutGuards car1Schema <$> car1Queries)
      <> (ParseQueryExprFails car1Schema <$> car1QueriesFails)

car1LexerTests :: TestItem
car1LexerTests =
  Group "car1" $
    LexQueryExpr car1Schema <$> car1Queries
