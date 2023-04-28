// https://stackoverflow.com/questions/73275346/how-to-stream-data-to-the-browser-with-google-cloud-functions-so-that-download-s/73370059#73370059?newreg=c0f1a7819e66426eb3a3a44e996bbbf0
// eslint-disable-next-line max-len
// Streaming is not possible from Firebase Functions because of the buffering implementation.

import { Milvus } from 'langchain/vectorstores/milvus'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { loadQARefineChain } from 'langchain/chains'
import { OpenAI } from 'langchain/llms/openai'
import { Request, Response } from 'express'
import { getErrorMessage } from './utils'

export async function get_similar_documents(
  question: string,
  collection_name: string,
  openApiKey: string
) {
  const url = `${process.env.MILVUS_HOST}:${process.env.MILVUS_PORT}`

  const vectorStore = await Milvus.fromExistingCollection(
    new OpenAIEmbeddings({ openAIApiKey: openApiKey }),
    {
      collectionName: collection_name,
      textField: 'otext',
      url,
    }
  )

  let response
  try {
    response = await vectorStore.similaritySearch(question, 4)
  } catch (error) {
    console.log('similaritySearch error:' + error)
    throw error
  }

  return response
}

export async function generate_answer(
  question: string,
  relevantDocs: any,
  model: OpenAI
) {
  const chain = loadQARefineChain(model)

  // Call the chain
  await chain.call({
    input_documents: relevantDocs,
    question,
  })
}

const questionAnswering = async (req: Request, res: Response) => {
  const { question, collection_name } = req.body
  const openApiKey = process.env.OPENAI_KEY

  res.setHeader('Content-Type', 'text/plain; charset=utf-8')

  if (!openApiKey) {
    res.status(400).json({
      message: 'Please provide an OpenAI API key.',
    })
  } else if (!question) {
    res.status(400).json({
      message: 'Invalid input. Please provide a question.',
    })
  } else {
    try {
      const similar_docs = await get_similar_documents(
        question,
        collection_name,
        openApiKey
      )
      console.log('similar_docs:' + similar_docs)

      const model = new OpenAI({
        streaming: true,
        temperature: 0,
        openAIApiKey: openApiKey,
        callbacks: [
          {
            handleLLMNewToken: (token: string) => {
              res.write(token)
            },
          },
        ],
      })
      await generate_answer(question, similar_docs, model)

      res.end()

      // res.status(200).send({
      //   status: 'success',
      //   data: answer,
      // })
    } catch (error) {
      res.status(500).json(getErrorMessage(error))
    }
  }
}

export { questionAnswering }
