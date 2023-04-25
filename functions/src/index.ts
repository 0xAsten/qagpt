// Setup Env variables for Milvus before running the code
// export COLLECTION_NAME=YOUR_COLLECTION_NAME_HERE
// export OPENAI_API_KEY=YOUR_OPEN_API_HERE
// export MILVUS_URL=YOUR_MILVUS_URL_HERE
// for example http://localhost:19530
import * as functions from 'firebase-functions'
import { defineString } from 'firebase-functions/params'
// import { MilvusClient } from '@zilliz/milvus2-sdk-node'
import { Milvus } from 'langchain/vectorstores/milvus'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { loadQARefineChain } from 'langchain/chains'
import { OpenAI } from 'langchain/llms/openai'

// const MILVUS_HOST = process.env.MILVUS_HOST || '127.0.0.1'
// const MILVUS_PORT = process.env.MILVUS_PORT || '19530'
// const milvus = new MilvusClient(`http://${MILVUS_HOST}:${MILVUS_PORT}`)
// const COLLECTION_NAME = process.env.COLLECTION_NAME || 'your_collection_name'

// const collection_name = defineString('COLLECTION_NAME', {
//   default: 'Cairo1',
//   description: 'The name of the collection to use in Milvus',
// })

async function get_similar_documents(
  question: string,
  collection_name: string
) {
  const vectorStore = await Milvus.fromExistingCollection(
    new OpenAIEmbeddings(),
    {
      collectionName: collection_name,
      textField: 'otext',
    }
  )

  const response = await vectorStore.similaritySearch(question, 4)

  return response
}

export async function generate_answer(
  question: string,
  collection_name: string
) {
  // Use GPT-4 to generate an answer based on the question and similar_docs
  // const answer = your_gpt_model_generate_answer_code_here
  const relevantDocs = await get_similar_documents(question, collection_name)

  const model = new OpenAI({ temperature: 0 })
  const chain = loadQARefineChain(model)

  // Call the chain
  const res = await chain.call({
    input_documents: relevantDocs,
    question,
  })
  // console.log({ res })
  return res
}

export const question_answering = functions.https.onRequest(
  async (request, response) => {
    const { question, collection_name } = request.body

    if (question) {
      const answer = await generate_answer(question, collection_name)
      response.status(200).send(answer)
    } else {
      response
        .status(400)
        .send({ error: 'Invalid input. Please provide a question.' })
    }
  }
)
