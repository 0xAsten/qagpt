// Setup Env variables for Milvus before running the code
// export COLLECTION_NAME=YOUR_COLLECTION_NAME_HERE
// export OPENAI_API_KEY=YOUR_OPEN_API_HERE
// export MILVUS_URL=YOUR_MILVUS_URL_HERE
// for example http://localhost:19530
import * as functions from 'firebase-functions'
import { defineString, defineSecret } from 'firebase-functions/params'
// import { MilvusClient } from '@zilliz/milvus2-sdk-node'
import { Milvus } from 'langchain/vectorstores/milvus'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { loadQARefineChain } from 'langchain/chains'
import { OpenAI } from 'langchain/llms/openai'
import { SecretParam } from 'firebase-functions/lib/params/types'

// const MILVUS_HOST = process.env.MILVUS_HOST || '127.0.0.1'
// const MILVUS_PORT = process.env.MILVUS_PORT || '19530'
// const milvus = new MilvusClient(`http://${MILVUS_HOST}:${MILVUS_PORT}`)
// const COLLECTION_NAME = process.env.COLLECTION_NAME || 'your_collection_name'

// const collection_name = defineString('COLLECTION_NAME', {
//   default: 'Cairo1',
//   description: 'The name of the collection to use in Milvus',
// })

const openApiKey = defineSecret('OPENAI_API_KEY')
const milvus_host = defineString('MILVUS_HOST', {
  // default: '127.0.0.1',
  description: 'The host of the Milvus server',
})
const milvus_port = defineString('MILVUS_PORT', {
  default: '19530',
  description: 'The port of the Milvus server',
})

async function get_similar_documents(
  question: string,
  collection_name: string,
  openApiKey: SecretParam
) {
  const url = `http://${milvus_host.value()}:${milvus_port.value()}`
  console.log(url)

  const vectorStore = await Milvus.fromExistingCollection(
    new OpenAIEmbeddings({ openAIApiKey: openApiKey.value() }),
    {
      collectionName: collection_name,
      textField: 'otext',
      url,
    }
  )

  const response = await vectorStore.similaritySearch(question, 4)

  return response
}

export async function generate_answer(question: string, relevantDocs: any) {
  // Use GPT-4 to generate an answer based on the question and similar_docs
  // const answer = your_gpt_model_generate_answer_code_here
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

export const question_answering = functions
  .runWith({ secrets: [openApiKey] })
  .https.onRequest(async (request, response) => {
    const { question, collection_name } = request.body

    if (question) {
      const similar_docs = await get_similar_documents(
        question,
        collection_name,
        openApiKey
      )
      const answer = await generate_answer(question, similar_docs)
      response.status(200).send(answer)
    } else {
      response
        .status(400)
        .send({ error: 'Invalid input. Please provide a question.' })
    }
  })
