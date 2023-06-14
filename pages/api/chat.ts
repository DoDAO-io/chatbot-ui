import { NextApiRequest, NextApiResponse } from 'next';

import { DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIError, OpenAIStream } from '@/utils/server';
import { ChatOpenAI } from 'langchain/chat_models';
import { CallbackManager } from 'langchain/callbacks';
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
import { uuid } from 'uuidv4';

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

    const { model, messages, prompt, temperature } = req.body as ChatBody;
    const encoding = encoding_for_model(model.id as TiktokenModel);

    // Embed the user's intent and query the Pinecone index
    const embedder = new OpenAIEmbeddings();

    const embeddings = await embedder.embedQuery(messages[messages.length-1].content);

    console.log('embeddings', embeddings.length);
    const matches = await getMatchesFromEmbeddings(embeddings, pinecone!, 3);

    console.log('matches', matches.length);
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

      let documentTokens = 0;
      if (fullDocuments) {
          for (let doc of fullDocuments) {
              documentTokens += encoding.encode(doc).length;
          }
      }
    
    if(documentTokens > 14000){
      const summary = await summarizeLongDocument(
          fullDocuments!.join('\n'),
          messages[messages.length-1].content,
          () => {
            console.log('onSummaryDone');
          },
        );
    }

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
    
   
    
    const promptToSend = await promptQA.format({
      summaries: fullDocuments!.join('\n'),
      question: prompt,
      conversationHistory: messages.map((m) => m.content),
      urls,
    });
    

    const prompt_tokens = encoding.encode(promptToSend);

    let tokenCount = prompt_tokens.length;
    let conversationHistory: Message[] = [];

   
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const tokens = encoding.encode(message.content);

      if (tokenCount + tokens.length + 1000 > model.tokenLimit) {
        break;
      }
      tokenCount += tokens.length;
      conversationHistory = [message, ...conversationHistory];
    }

    encoding.free();

    const promptTemplate = new PromptTemplate({
      template: templates.qaTemplate,
      inputVariables: ['summaries', 'question', 'conversationHistory', 'urls'],
    });


    const key = process.env.OPENAI_API_KEY!;
    const chat = new ChatOpenAI({
      streaming: true,
      verbose: true,
      modelName: 'gpt-3.5-turbo-16k',
      callbackManager: CallbackManager.fromHandlers({
        async handleLLMNewToken(token) {
          console.log(token);
          res.write(token);
        },
        async handleLLMEnd(result) {
          res.end();
        },
      }),
    });
    const chain = new LLMChain({
      prompt: promptTemplate,
      llm: chat,
    });

    await chain.call({
      summaries: fullDocuments!.join('\n'),
      question: promptToSend,
      conversationHistory,
      urls,
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
