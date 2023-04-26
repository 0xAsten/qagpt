import { get_similar_documents, generate_answer } from '../qaController'

describe('getSimiliarDocs', () => {
  it('should get similar documents', async () => {
    const openAPIKey = process.env.OPENAI_API_KEY
    expect(openAPIKey).toBeDefined()

    const similarDocs = await get_similar_documents(
      'What is Cairo?',
      'Cairo1',
      openAPIKey!
    )
    expect(similarDocs).toBeDefined()
    // console.log(similarDocs)
  })
  it('should generate answer', async () => {
    const openAPIKey = process.env.OPENAI_API_KEY
    expect(openAPIKey).toBeDefined()

    const similarDocs = await get_similar_documents(
      'What is Cairo?',
      'Cairo1',
      openAPIKey!
    )

    const answer = await generate_answer(
      'What is Cairo?',
      similarDocs,
      openAPIKey!
    )
    expect(answer).toBeDefined()
    console.log(answer)
  })
})
