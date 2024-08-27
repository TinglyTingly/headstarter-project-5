import {
  Message as VercelChatMessage,
  StreamingTextResponse,
  createStreamDataTransformer,
} from "ai";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { HttpResponseOutputParser } from "langchain/output_parsers";

import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { CharacterTextSplitter } from "langchain/text_splitter";

const loader = new JSONLoader("src/data/nestedReviews.json");

export const dynamic = "force-dynamic";

/**
 * Basic memory formatter that stringifies and passes
 * message history directly into the model.
 */
const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

const TEMPLATE = `Answer the user's questions based only on the following context. If the answer is not in the context, reply politely that you do not have that information available.:
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;

export async function POST(req: Request) {
  try {
    // Extract the `messages` from the body of the request
    const { messages } = await req.json();

    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);

    const currentMessageContent = messages[messages.length - 1].content;

    // const docs = await loader.load();

    // load a JSON object
    const textSplitter = new CharacterTextSplitter();
    const docs = await textSplitter.createDocuments([
      JSON.stringify({
        professors: [
          {
            id: 1,
            name: "Dr. Samantha Lee",
            department: "Biology",
            institution: "Green Valley University",
            overallRating: 4.2,
            wouldTakeAgainPercent: 88,
            levelOfDifficulty: 3.7,
            reviews: [
              {
                date: "2024-03-15",
                course: "BIO 201",
                rating: 4,
                difficulty: 4,
                comment:
                  "Dr. Lee is passionate about biology and it shows in her lectures.",
              },
            ],
          },
          {
            id: 2,
            name: "Prof. Michael Chen",
            department: "Computer Science",
            institution: "Tech Institute",
            overallRating: 3.9,
            wouldTakeAgainPercent: 75,
            levelOfDifficulty: 4.2,
            reviews: [
              {
                date: "2024-02-28",
                course: "CS 301",
                rating: 4,
                difficulty: 5,
                comment:
                  "Challenging course, but Prof. Chen is always willing to help.",
              },
            ],
          },
          {
            id: 3,
            name: "Dr. Emily Johnson",
            department: "Psychology",
            institution: "Mindful State University",
            overallRating: 4.7,
            wouldTakeAgainPercent: 92,
            levelOfDifficulty: 3.2,
            reviews: [
              {
                date: "2024-03-10",
                course: "PSYCH 101",
                rating: 5,
                difficulty: 3,
                comment:
                  "Dr. Johnson makes psychology fascinating and accessible.",
              },
            ],
          },
          {
            id: 4,
            name: "Prof. David Williams",
            department: "History",
            institution: "Heritage College",
            overallRating: 3.5,
            wouldTakeAgainPercent: 65,
            levelOfDifficulty: 3.8,
            reviews: [
              {
                date: "2024-03-05",
                course: "HIST 202",
                rating: 3,
                difficulty: 4,
                comment: "Lectures can be dry, but the content is interesting.",
              },
            ],
          },
          {
            id: 5,
            name: "Dr. Sarah Thompson",
            department: "Chemistry",
            institution: "Molecular University",
            overallRating: 4.4,
            wouldTakeAgainPercent: 85,
            levelOfDifficulty: 4.0,
            reviews: [
              {
                date: "2024-03-18",
                course: "CHEM 301",
                rating: 4,
                difficulty: 4,
                comment: "Tough but fair. Dr. Thompson really knows her stuff.",
              },
            ],
          },
          {
            id: 6,
            name: "Prof. Robert Garcia",
            department: "Economics",
            institution: "Finance State University",
            overallRating: 3.8,
            wouldTakeAgainPercent: 72,
            levelOfDifficulty: 3.9,
            reviews: [
              {
                date: "2024-02-20",
                course: "ECON 201",
                rating: 4,
                difficulty: 4,
                comment: "Clear explanations of complex economic concepts.",
              },
            ],
          },
          {
            id: 7,
            name: "Dr. Lisa Brown",
            department: "English Literature",
            institution: "Wordsmith College",
            overallRating: 4.6,
            wouldTakeAgainPercent: 90,
            levelOfDifficulty: 3.3,
            reviews: [
              {
                date: "2024-03-12",
                course: "ENG 305",
                rating: 5,
                difficulty: 3,
                comment: "Dr. Brown's passion for literature is contagious.",
              },
            ],
          },
          {
            id: 8,
            name: "Prof. James Wilson",
            department: "Physics",
            institution: "Quantum State University",
            overallRating: 3.7,
            wouldTakeAgainPercent: 68,
            levelOfDifficulty: 4.5,
            reviews: [
              {
                date: "2024-03-08",
                course: "PHYS 202",
                rating: 3,
                difficulty: 5,
                comment:
                  "Brilliant physicist, but lectures can be hard to follow.",
              },
            ],
          },
          {
            id: 9,
            name: "Dr. Maria Rodriguez",
            department: "Sociology",
            institution: "Social Sciences Institute",
            overallRating: 4.3,
            wouldTakeAgainPercent: 86,
            levelOfDifficulty: 3.5,
            reviews: [
              {
                date: "2024-03-14",
                course: "SOC 101",
                rating: 4,
                difficulty: 3,
                comment:
                  "Engaging discussions and relevant real-world examples.",
              },
            ],
          },
          {
            id: 10,
            name: "Prof. Thomas Anderson",
            department: "Mathematics",
            institution: "Calculus College",
            overallRating: 3.9,
            wouldTakeAgainPercent: 74,
            levelOfDifficulty: 4.2,
            reviews: [
              {
                date: "2024-02-25",
                course: "MATH 301",
                rating: 4,
                difficulty: 4,
                comment:
                  "Challenging course, but Prof. Anderson is always available for help.",
              },
            ],
          },
          {
            id: 11,
            name: "Dr. Jennifer Lee",
            department: "Political Science",
            institution: "Governance University",
            overallRating: 4.5,
            wouldTakeAgainPercent: 89,
            levelOfDifficulty: 3.6,
            reviews: [
              {
                date: "2024-03-17",
                course: "POLI 202",
                rating: 5,
                difficulty: 4,
                comment:
                  "Dr. Lee brings political theories to life with current events.",
              },
            ],
          },
          {
            id: 12,
            name: "Prof. Daniel Kim",
            department: "Art History",
            institution: "Creative Arts College",
            overallRating: 4.1,
            wouldTakeAgainPercent: 82,
            levelOfDifficulty: 3.3,
            reviews: [
              {
                date: "2024-03-09",
                course: "ART 305",
                rating: 4,
                difficulty: 3,
                comment:
                  "Prof. Kim's enthusiasm for art history is infectious.",
              },
            ],
          },
          {
            id: 13,
            name: "Dr. Rachel Green",
            department: "Environmental Science",
            institution: "Eco State University",
            overallRating: 4.4,
            wouldTakeAgainPercent: 87,
            levelOfDifficulty: 3.8,
            reviews: [
              {
                date: "2024-03-11",
                course: "ENV 201",
                rating: 4,
                difficulty: 4,
                comment:
                  "Challenging but rewarding. Dr. Green is passionate about sustainability.",
              },
            ],
          },
          {
            id: 14,
            name: "Prof. William Taylor",
            department: "Philosophy",
            institution: "Thinkers College",
            overallRating: 3.8,
            wouldTakeAgainPercent: 71,
            levelOfDifficulty: 4.1,
            reviews: [
              {
                date: "2024-02-22",
                course: "PHIL 101",
                rating: 4,
                difficulty: 4,
                comment:
                  "Thought-provoking lectures, but be prepared for heavy reading.",
              },
            ],
          },
          {
            id: 15,
            name: "Dr. Karen Martinez",
            department: "Neuroscience",
            institution: "Brain Research University",
            overallRating: 4.6,
            wouldTakeAgainPercent: 91,
            levelOfDifficulty: 4.0,
            reviews: [
              {
                date: "2024-03-16",
                course: "NEUR 301",
                rating: 5,
                difficulty: 4,
                comment:
                  "Dr. Martinez makes complex neuroscience concepts understandable.",
              },
            ],
          },
          {
            id: 16,
            name: "Prof. Christopher Lee",
            department: "Music Theory",
            institution: "Harmony Conservatory",
            overallRating: 4.2,
            wouldTakeAgainPercent: 84,
            levelOfDifficulty: 3.5,
            reviews: [
              {
                date: "2024-03-07",
                course: "MUS 202",
                rating: 4,
                difficulty: 3,
                comment:
                  "Prof. Lee's passion for music theory is evident in every lecture.",
              },
            ],
          },
          {
            id: 17,
            name: "Dr. Amanda White",
            department: "Anthropology",
            institution: "Cultural Studies University",
            overallRating: 4.3,
            wouldTakeAgainPercent: 86,
            levelOfDifficulty: 3.7,
            reviews: [
              {
                date: "2024-03-13",
                course: "ANTH 101",
                rating: 4,
                difficulty: 4,
                comment:
                  "Fascinating course content and engaging class discussions.",
              },
            ],
          },
          {
            id: 18,
            name: "Prof. Richard Brown",
            department: "Mechanical Engineering",
            institution: "Tech Innovation Institute",
            overallRating: 3.9,
            wouldTakeAgainPercent: 75,
            levelOfDifficulty: 4.3,
            reviews: [
              {
                date: "2024-02-26",
                course: "MECH 301",
                rating: 4,
                difficulty: 5,
                comment:
                  "Challenging course, but Prof. Brown is knowledgeable and helpful.",
              },
            ],
          },
          {
            id: 19,
            name: "Dr. Elizabeth Chen",
            department: "Genetics",
            institution: "Genomic State University",
            overallRating: 4.5,
            wouldTakeAgainPercent: 88,
            levelOfDifficulty: 4.0,
            reviews: [
              {
                date: "2024-03-19",
                course: "GEN 202",
                rating: 5,
                difficulty: 4,
                comment:
                  "Dr. Chen explains complex genetic concepts clearly and engagingly.",
              },
            ],
          },
          {
            id: 20,
            name: "Prof. Mark Johnson",
            department: "Marketing",
            institution: "Business Strategy College",
            overallRating: 4.0,
            wouldTakeAgainPercent: 79,
            levelOfDifficulty: 3.6,
            reviews: [
              {
                date: "2024-03-04",
                course: "MKT 301",
                rating: 4,
                difficulty: 3,
                comment:
                  "Prof. Johnson brings real-world marketing experience to the classroom.",
              },
            ],
          },
          {
            id: 21,
            name: "Dr. Laura Thompson",
            department: "Linguistics",
            institution: "Language Arts University",
            overallRating: 4.4,
            wouldTakeAgainPercent: 87,
            levelOfDifficulty: 3.8,
            reviews: [
              {
                date: "2024-03-20",
                course: "LING 201",
                rating: 4,
                difficulty: 4,
                comment:
                  "Dr. Thompson's enthusiasm for linguistics is contagious.",
              },
            ],
          },
          {
            id: 22,
            name: "Prof. Andrew Davis",
            department: "Astronomy",
            institution: "Celestial Sciences Institute",
            overallRating: 4.2,
            wouldTakeAgainPercent: 83,
            levelOfDifficulty: 3.9,
            reviews: [
              {
                date: "2024-03-01",
                course: "ASTR 101",
                rating: 4,
                difficulty: 4,
                comment: "Prof. Davis makes astronomy accessible and exciting.",
              },
            ],
          },
          {
            id: 23,
            name: "Dr. Sophia Rodriguez",
            department: "Women's Studies",
            institution: "Equality State University",
            overallRating: 4.6,
            wouldTakeAgainPercent: 91,
            levelOfDifficulty: 3.5,
            reviews: [
              {
                date: "2024-03-18",
                course: "WMST 201",
                rating: 5,
                difficulty: 3,
                comment:
                  "Dr. Rodriguez facilitates thought-provoking discussions on gender issues.",
              },
            ],
          },
          {
            id: 24,
            name: "Prof. Kevin Park",
            department: "Film Studies",
            institution: "Cinema Arts College",
            overallRating: 4.1,
            wouldTakeAgainPercent: 80,
            levelOfDifficulty: 3.4,
            reviews: [
              {
                date: "2024-02-27",
                course: "FILM 301",
                rating: 4,
                difficulty: 3,
                comment:
                  "Prof. Park's lectures are as entertaining as they are informative.",
              },
            ],
          },
          {
            id: 25,
            name: "Dr. Olivia Brown",
            department: "Nutrition Science",
            institution: "Health Sciences University",
            overallRating: 4.3,
            wouldTakeAgainPercent: 85,
            levelOfDifficulty: 3.7,
            reviews: [
              {
                date: "2024-03-14",
                course: "NUTR 202",
                rating: 4,
                difficulty: 4,
                comment:
                  "Dr. Brown provides practical insights into nutrition and health.",
              },
            ],
          },
        ],
      }),
    ]);

    const prompt = PromptTemplate.fromTemplate(TEMPLATE);

    const model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: "gpt-3.5-turbo",
      temperature: 0.2,
      streaming: true,
      verbose: true,
    });

    /**
     * Chat models stream message chunks rather than bytes, so this
     * output parser handles serialization and encoding.
     */
    const parser = new HttpResponseOutputParser();

    const chain = RunnableSequence.from([
      {
        question: (input) => input.question,
        chat_history: (input) => input.chat_history,
        context: () => formatDocumentsAsString(docs),
      },
      prompt,
      model,
      parser,
    ]);

    // Convert the response into a friendly text-stream
    const stream = await chain.stream({
      chat_history: formattedPreviousMessages.join("\n"),
      question: currentMessageContent,
    });

    // Respond with the stream
    return new StreamingTextResponse(
      stream.pipeThrough(createStreamDataTransformer())
    );
  } catch (e: any) {
    return Response.json({ error: e.message }, { status: e.status ?? 500 });
  }
}
