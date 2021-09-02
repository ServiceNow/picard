{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.ProductCatalog where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

productCatalogSchema :: SQLSchema
productCatalogSchema =
  let columnNames = HashMap.fromList [("15", "previous_entry_id"), ("7", "date_of_publication"), ("25", "width"), ("28", "attribute_id"), ("13", "catalog_level_number"), ("14", "parent_entry_id"), ("22", "capacity"), ("19", "price_in_dollars"), ("29", "attribute_value"), ("12", "catalog_entry_id"), ("17", "catalog_entry_name"), ("1", "attribute_id"), ("23", "length"), ("18", "product_stock_number"), ("4", "catalog_id"), ("26", "catalog_entry_id"), ("16", "next_entry_id"), ("2", "attribute_name"), ("20", "price_in_euros"), ("5", "catalog_name"), ("27", "catalog_level_number"), ("8", "date_of_latest_revision"), ("11", "catalog_level_name"), ("3", "attribute_data_type"), ("21", "price_in_pounds"), ("24", "height"), ("6", "catalog_publisher"), ("9", "catalog_level_number"), ("10", "catalog_id")]
      tableNames = HashMap.fromList [("0", "Attribute_Definitions"), ("1", "Catalogs"), ("4", "Catalog_Contents_Additional_Attributes"), ("2", "Catalog_Structure"), ("3", "Catalog_Contents")]
      columnToTable = HashMap.fromList [("15", "3"), ("7", "1"), ("25", "3"), ("28", "4"), ("13", "3"), ("14", "3"), ("22", "3"), ("19", "3"), ("29", "4"), ("12", "3"), ("17", "3"), ("1", "0"), ("23", "3"), ("18", "3"), ("4", "1"), ("26", "4"), ("16", "3"), ("2", "0"), ("20", "3"), ("5", "1"), ("27", "4"), ("8", "1"), ("11", "2"), ("3", "0"), ("21", "3"), ("24", "3"), ("6", "1"), ("9", "2"), ("10", "2")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3"]), ("1", ["4", "5", "6", "7", "8"]), ("4", ["26", "27", "28", "29"]), ("2", ["9", "10", "11"]), ("3", ["12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"])]
      foreignKeys = HashMap.fromList [("13", "9"), ("26", "12"), ("27", "9"), ("10", "4")]
      foreignKeysTables = HashMap.fromList [("4", ["2", "3"]), ("2", ["1"]), ("3", ["2"])]
      primaryKeys = ["1", "4", "9", "12"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

productCatalogQueries :: [Text.Text]
productCatalogQueries =
  [ "select catalog_entry_name from catalog_contents as T1",
    "select distinct (T1.catalog_entry_name) from catalog_contents as T1",
    "select distinct(catalog_entry_name) from catalog_contents",
    "select distinct(catalog_publisher) from catalogs where catalog_publisher like \"%murray%\""
  ]

productCatalogQueriesFails :: [Text.Text]
productCatalogQueriesFails = []

productCatalogParserTests :: TestItem
productCatalogParserTests =
  Group "productCatalog" $
    (ParseQueryExprWithGuards productCatalogSchema <$> productCatalogQueries)
      <> (ParseQueryExprWithoutGuards productCatalogSchema <$> productCatalogQueries)
      <> (ParseQueryExprFails productCatalogSchema <$> productCatalogQueriesFails)

productCatalogLexerTests :: TestItem
productCatalogLexerTests =
  Group "productCatalog" $
    LexQueryExpr productCatalogSchema <$> productCatalogQueries
