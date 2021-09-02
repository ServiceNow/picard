module Language.SQL.SpiderSQL.TestItem where

import qualified Data.Text as Text
import Picard.Types (SQLSchema)

data TestItem
  = Group String [TestItem]
  | LexQueryExpr SQLSchema Text.Text
  | ParseQueryExprWithoutGuards SQLSchema Text.Text
  | ParseQueryExprWithGuards SQLSchema Text.Text
  | ParseQueryExprFails SQLSchema Text.Text
  deriving stock (Eq, Show)
