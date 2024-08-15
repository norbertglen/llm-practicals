import "reflect-metadata";
import env from 'dotenv';
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { ChatOpenAI } from "@langchain/openai";
import { createSqlQueryChain } from "langchain/chains/sql_db";
import { QuerySqlTool } from "langchain/tools/sql";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";


env.config()

const datasource = new DataSource({
  type: "mysql",
  host: "localhost",
  port: 3306,
  username: "root",
  password: "P@ssword123",
  database: "dala",
});
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});
// console.log(db.allTables.map((t) => t.tableName));
const llm = new ChatOpenAI({ model: "gpt-4", temperature: 0 })
const executeQuery = new QuerySqlTool(db)
const writeQuery = await createSqlQueryChain({
  llm,
  db,
  dialect: "mysql"
})

const answerPrompt = PromptTemplate.fromTemplate(`Given the following user question, 
  corresponding SQL query, and SQL result, answer the user question.
  Question: {question}
  SQL Query: {query}
  SQL Result: {result}
  Answer: `)

const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser())
const chain = RunnableSequence.from([
  RunnablePassthrough.assign({query: writeQuery}).assign({result: (i) => executeQuery.invoke(i.query)}),
  answerChain
])
console.log(await chain.invoke({
  question: "Which users pay over 30000 per month? no limit."
}))