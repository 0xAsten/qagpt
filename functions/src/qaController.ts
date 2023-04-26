import * as functions from 'firebase-functions'
import { Milvus } from 'langchain/vectorstores/milvus'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { loadQARefineChain } from 'langchain/chains'
import { OpenAI } from 'langchain/llms/openai'
import { Request, Response } from 'express'
import { getErrorMessage } from './utils'
import { PassThrough } from 'stream'

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

  const des = await vectorStore.client.describeCollection({ collection_name })
  console.log(des)

  const response = await vectorStore.similaritySearch(question, 4)

  return response
}

export async function generate_answer(
  question: string,
  relevantDocs: any,
  openApiKey: string,
  passThrough: PassThrough
) {
  // Use GPT-4 to generate an answer based on the question and similar_docs
  // const answer = your_gpt_model_generate_answer_code_here
  const model = new OpenAI({
    streaming: true,
    temperature: 0,
    openAIApiKey: openApiKey,
    callbacks: [
      {
        handleLLMNewToken(token: string) {
          passThrough.write(token)
        },
      },
    ],
  })
  const chain = loadQARefineChain(model)

  // Call the chain
  await chain.call({
    input_documents: relevantDocs,
    question,
  })
}

export const questionAnswering = functions
  .runWith({ secrets: ['OPENAI_KEY'] })
  .https.onRequest(async (req: Request, res: Response) => {
    const { question, collection_name } = req.body
    const openApiKey = process.env.OPENAI_KEY

    const passThrough = new PassThrough()
    passThrough.pipe(res)

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

        await generate_answer(question, similar_docs, openApiKey, passThrough)

        passThrough.end()

        // res.status(200).send({
        //   status: 'success',
        //   data: answer,
        // })
      } catch (error) {
        res.status(500).json(getErrorMessage(error))
      }
    }
  })
