import { get_similar_documents } from '../src/qaController'

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
    console.log(similarDocs)
  }, 20000)
})
