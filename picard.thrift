typedef string ColumnId

typedef string TableId

typedef string DBId

typedef i64 Token

typedef list<Token> InputIds

typedef i64 BatchId

struct SQLSchema {
    1: map<ColumnId,string> columnNames,
    2: map<TableId,string> tableNames,
    3: map<ColumnId,TableId> columnToTable,
    4: map<TableId,list<ColumnId>> tableToColumns,
    5: map<ColumnId,ColumnId> foreignKeys,
    6: map<TableId,list<TableId>> foreignKeysTables,
    7: list<ColumnId> primaryKeys,
}

exception RegisterSQLSchemaException {
    1: DBId dbId,
    2: string message,
}

exception RegisterTokenizerException {
    1: string message,
}

struct TokenizerNotRegisteredException {
    1: string message,
}

struct TokenizerPrefixException {
    1: string message,
}

struct ModeException {
    1: string message,
}

union FeedFatalException {
    1: TokenizerNotRegisteredException tokenizerNotRegisteredException,
    2: TokenizerPrefixException tokenizerPrefixException,
    3: ModeException modeException
}

exception FeedException {
    1: FeedFatalException feedFatalException,
}

struct FeedParseFailure {
    1: string input,
    2: list<string> contexts,
    3: string description,
}

struct FeedTimeoutFailure {
    1: string message,
}

struct FeedPartialSuccess {}

struct FeedCompleteSuccess {
    1: string leftover,
}

union FeedResult {
    1: FeedParseFailure feedParseFailure,
    2: FeedTimeoutFailure feedTimeoutFailure,
    3: FeedPartialSuccess feedPartialSuccess,
    4: FeedCompleteSuccess feedCompleteSuccess,
}

struct BatchFeedResult {
    1: BatchId batchId,
    2: Token topToken,
    3: FeedResult feedResult,
}

enum Mode {
    LEXING = 1,
    PARSING_WITHOUT_GUARDS = 2,
    PARSING_WITH_GUARDS = 3,
}

service Picard {
   void registerSQLSchema(1:DBId dbId, 2:SQLSchema sqlSchema) throws (1:RegisterSQLSchemaException fail),
   void registerTokenizer(1:string jsonConfig) throws (1:RegisterTokenizerException fail),
   FeedResult feed(1:InputIds inputIds, 2:Token token, 3:Mode mode) throws (1:FeedException fail),
   list<BatchFeedResult> batchFeed(1:list<InputIds> inputIds, 2:list<list<Token>> topTokens, 3:Mode mode) throws (1:FeedException fail),
}
