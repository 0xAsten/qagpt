import * as functions from 'firebase-functions'
import * as express from 'express'
import * as cors from 'cors'
import { questionAnswering } from './qaController'

const app = express()

app.use(cors())

app.get('/', (req, res) => res.status(200).send('Hey, there!'))
app.post('/qa', questionAnswering)

exports.app = functions
  .runWith({ secrets: ['OPENAI_KEY'] })
  .https.onRequest(app)
