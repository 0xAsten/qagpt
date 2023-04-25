import * as functions from 'firebase-functions'
import * as express from 'express'
import * as cors from 'cors'
import { question_answering } from './qaController'

const app = express()

app.use(cors())

app.get('/', (req, res) => res.status(200).send('Hey, there!'))
app.post('/qa', question_answering)

exports.app = functions.https.onRequest(app)
