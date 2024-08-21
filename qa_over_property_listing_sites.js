import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import env from 'dotenv';
import { formatDocumentsAsString } from "langchain/util/document";
import {
    RunnableSequence,
    RunnablePassthrough,
} from "@langchain/core/runnables";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

(async () => {
    env.config()
    // scrap and QA properties listed buyrentkenya.com
    const getDocuments = (url) => new Promise((resolve, reject) => {
        const loader = new CheerioWebBaseLoader(url)
        loader.load().then(docs => resolve(docs)).catch(() => resolve([]));
    })
    const docs = await Promise.all(Array.from(Array(50).keys()).map(page => getDocuments(`https://www.buyrentkenya.com/property-for-rent/?sort=latest${page === 0 ? "" : "&page=" + page}`)))

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })
    const splits = await textSplitter.splitDocuments(...docs)
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings())
    const retriever = vectorStore.asRetriever({ k: 6, searchType: "similarity" });
    const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", temperature: 0 })

    const template = `
        You are a real estate search assistant helping users find properties.
        The context provided has information scrapped from a property listing website about different properties, their features, location and prices etc. 
        Use the pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:`;

    const customRagPrompt = PromptTemplate.fromTemplate(template);

    const ragChain = await createStuffDocumentsChain({
        llm,
        prompt: customRagPrompt,
        outputParser: new StringOutputParser(),
    });
    const question = `What rentals would you recommend for me in Kilimani. 
    My budget range is within 50000 and 100000?
    Also let me know of any highlight features about the unit
    `
    const context = await retriever.invoke(question);

    const result = await ragChain.invoke({
        question,
        context,
    });
    console.log(result);
})();
