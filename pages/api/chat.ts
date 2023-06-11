import { NextApiRequest, NextApiResponse } from 'next';

import { DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIError, OpenAIStream } from '@/utils/server';

import { ChatBody, Message } from '@/types/chat';

import { Metadata, getMatchesFromEmbeddings } from '@/pages/api/matches';
import { summarizeLongDocument } from '@/pages/api/summarizer';
import { templates } from '@/pages/api/templates';

import { TiktokenModel, encoding_for_model } from '@dqbd/tiktoken';
import { PineconeClient } from '@pinecone-database/pinecone';
import { LLMChain } from 'langchain/chains';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import { PromptTemplate } from 'langchain/prompts';


const llm = new OpenAI({});
let pinecone: PineconeClient | null = null;

const initPineconeClient = async () => {
  pinecone = new PineconeClient();
  await pinecone.init({
    environment: process.env.PINECONE_ENVIRONMENT!,
    apiKey: process.env.PINECONE_API_KEY!,
  });
};

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  try {
    if (!pinecone) {
      await initPineconeClient();
    }

    console.log('body', req.body);

    const { model, messages, prompt, temperature } = req.body as ChatBody;

    console.log('messages', messages);
    const encoding = encoding_for_model(model.id as TiktokenModel);

    // Build an LLM chain that will improve the user prompt
    const inquiryChain = new LLMChain({
      llm,
      prompt: new PromptTemplate({
        template: templates.inquiryTemplate,
        inputVariables: ['userPrompt', 'conversationHistory'],
      }),
    });
    const inquiryChainResult = await inquiryChain.call({
      userPrompt: prompt,
      conversationHistory: messages.map((m) => m.content),
    });
    const inquiry = inquiryChainResult.text;

    // Embed the user's intent and query the Pinecone index
    const embedder = new OpenAIEmbeddings();

    const embeddings = await embedder.embedQuery(inquiry);

    console.log('embeddings', embeddings.length);
    const matches = await getMatchesFromEmbeddings(embeddings, pinecone!, 3);

    console.log('matches', matches.length);

    // const urls = docs && Array.from(new Set(docs.map(doc => doc.metadata.url)))

    const urls =
      matches &&
      Array.from(
        new Set(
          matches.map((match) => {
            const metadata = match.metadata as Metadata;
            const { url } = metadata;
            return url;
          }),
        ),
      );

    console.log(urls);

    const fullDocuments =
      matches &&
      Array.from(
        matches.reduce((map, match) => {
          const metadata = match.metadata as Metadata;
          const { text, url } = metadata;
          if (!map.has(url)) {
            map.set(url, text);
          }
          return map;
        }, new Map()),
      ).map(([_, text]) => text);

    const chunkedDocs =
      matches &&
      Array.from(
        new Set(
          matches.map((match) => {
            const metadata = match.metadata as Metadata;
            const { chunk } = metadata;
            return chunk;
          }),
        ),
      );

    let temperatureToUse = temperature;
    if (temperatureToUse == null) {
      temperatureToUse = DEFAULT_TEMPERATURE;
    }

    const promptQA = PromptTemplate.fromTemplate(templates.qaTemplate);

    const summary = await summarizeLongDocument(
      fullDocuments!.join('\n'),
      inquiry,
      () => {
        console.log('onSummaryDone');
      },
    );

    const promptToSend = await promptQA.format({
      summaries: summary,
      question: prompt,
      conversationHistory: messages.map((m) => m.content),
      urls,
    });

    const prompt_tokens = encoding.encode(promptToSend);

    let tokenCount = prompt_tokens.length;
    let messagesToSend: Message[] = [];

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const tokens = encoding.encode(message.content);

      if (tokenCount + tokens.length + 1000 > model.tokenLimit) {
        break;
      }
      tokenCount += tokens.length;
      messagesToSend = [message, ...messagesToSend];
    }

    encoding.free();

    const key = process.env.OPENAI_API_KEY!;

    const stream: ReadableStream = await OpenAIStream(
      model,
      promptToSend,
      temperatureToUse,
      key,
      messagesToSend,
    );

    // Set appropriate headers for a stream
    res.setHeader('Content-Type', 'text/plain'); // Set your Content-Type based on your stream data type
    res.setHeader('Transfer-Encoding', 'chunked');

    // Get a reader from the stream
    const reader = stream.getReader();

    // Read the stream and write data to the response
    const readAndWrite = async () => {
      const { value, done } = await reader.read();
      if (done) {
        // If stream is finished, end the response
        res.end();
        return;
      }

      // If stream is not finished, write chunk and recursively read the next
      res.write(value);
      readAndWrite();
    };

    readAndWrite().catch((err) => {
      // Log error and send error status
      console.error(err);
      res.status(500).end('An error occurred while streaming data.');
    });
  } catch (error) {
    console.error(error);
    if (error instanceof OpenAIError) {
      return new Response('Error', { status: 500, statusText: error.message });
    } else {
      return new Response('Error', { status: 500 });
    }
  }
};

export default handler;
