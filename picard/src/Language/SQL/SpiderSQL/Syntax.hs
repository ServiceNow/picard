module Language.SQL.SpiderSQL.Syntax where

import Data.Hashable (Hashable)
import GHC.Generics (Generic)

data SpiderSQL = SpiderSQL
  { spiderSQLSelect :: Select,
    spiderSQLFrom :: From,
    spiderSQLWhere :: Maybe Cond,
    spiderSQLGroupBy :: [ColUnit],
    spiderSQLOrderBy :: Maybe OrderBy,
    spiderSQLHaving :: Maybe Cond,
    spiderSQLLimit :: Maybe Int,
    spiderSQLIntersect :: Maybe SpiderSQL,
    spiderSQLExcept :: Maybe SpiderSQL,
    spiderSQLUnion :: Maybe SpiderSQL
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data Select
  = Select [Agg]
  | SelectDistinct [Agg]
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data From = From
  { fromTableUnits :: [TableUnit],
    fromCond :: Maybe Cond
  }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data Cond
  = And Cond Cond
  | Or Cond Cond
  | Not Cond
  | Between ValUnit ValUnit ValUnit
  | Eq ValUnit ValUnit
  | Gt ValUnit ValUnit
  | Lt ValUnit ValUnit
  | Ge ValUnit ValUnit
  | Le ValUnit ValUnit
  | Ne ValUnit ValUnit
  | In ValUnit ValUnit
  | Like ValUnit ValUnit
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data ValUnit
  = Column Val
  | Minus Val Val
  | Plus Val Val
  | Times Val Val
  | Divide Val Val
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data Val
  = ValColUnit {columnValue :: ColUnit}
  | Number {numberValue :: Double}
  | ValString {stringValue :: String}
  | ValSQL {sqlValue :: SpiderSQL}
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data ColUnit
  = ColUnit
      { colUnitAggId :: Maybe AggType,
        colUnitTable :: Maybe (Either TableId Alias),
        colUnitColId :: ColumnId
      }
  | DistinctColUnit
      { distinctColUnitAggId :: Maybe AggType,
        distinctColUnitTable :: Maybe (Either TableId Alias),
        distinctColUnitColdId :: ColumnId
      }
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

newtype OrderBy = OrderBy [(ValUnit, OrderByOrder)]
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data OrderByOrder = Asc | Desc
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data Agg = Agg (Maybe AggType) ValUnit
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data TableUnit
  = TableUnitSQL SpiderSQL (Maybe Alias)
  | Table TableId (Maybe Alias)
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data AggType = Max | Min | Count | Sum | Avg
  deriving stock (Eq, Show, Generic)
  deriving anyclass (Hashable)

data ColumnId = Star | ColumnId {columnName :: String}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (Hashable)

newtype TableId = TableId {tableName :: String}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (Hashable)

newtype Alias = Alias {aliasName :: String}
  deriving stock (Eq, Ord, Show, Generic)
  deriving anyclass (Hashable)
