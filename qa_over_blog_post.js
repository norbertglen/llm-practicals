import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import env from 'dotenv';

(async () => {
    env.config()

    const loader = new CheerioWebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    const docs = await loader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })
    const splits = await textSplitter.splitDocuments(docs)
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings())
    const retriever = vectorStore.asRetriever({ k: 6, searchType: "similarity" });
    const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 })

    const retrievedDocs = await retriever.invoke(
        "What is task decomposition?"
    );
    const template = await ChatPromptTemplate.fromTemplate(`You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:`);

    const prompt = await template.format({ question: "What is task decomposition?", context: retrievedDocs });
    const result = await llm.invoke(prompt); 
    
    console.log(result?.content);
})();
