import { NextApiRequest, NextApiResponse } from "next";

import { TokenSourceRequestPayload } from "livekit-client";
import { AccessToken } from "livekit-server-sdk";
import { RoomConfiguration } from "@livekit/protocol";

const apiKey = process.env.LIVEKIT_API_KEY;
const apiSecret = process.env.LIVEKIT_API_SECRET;

type CallContext = {
  prompt?: string;
  provider?: string;
  model?: string;
  stt_model?: string;
  language?: string;
  voice?: string;
};

type TokenRequest = {
  room_name: string;
  participant_identity: string;
  participant_name?: string;
  participant_metadata?: string;
  participant_attributes?: Record<string, string>;
  room_config?: ReturnType<RoomConfiguration["toJson"]>;
  call_context?: CallContext;
};

// This route handler creates a token for a given room and participant
// it's compatible with LiveKit's TokenSourceEndpoint API
async function createToken(request: TokenRequest) {
  // Default call context — overridden by values in participant_metadata
  // sent from the frontend via tokenFetchOptions.participantMetadata.
  // The LiveKit SDK converts participantMetadata → participant_metadata in
  // the POST body, which is a known SDK field that IS actually transmitted.
  const defaultCallContext: CallContext = {
    prompt: "You are a helpful assistant.",
    provider: "groq",
    model: "llama-3.3-70b-versatile",
    stt_model: "nova-2-general",
    language: "hi",
    voice: "cgSgspJ2msm6clMCkdW9", // Jessica
  };

  console.log(request.call_context, "call_context")

  // Parse call_context from participant_metadata JSON string sent by the SDK
  let incomingContext: Partial<CallContext> = {};
  if (request.participant_metadata) {
    try {
      incomingContext = JSON.parse(request.participant_metadata);
      console.log("[TOKEN] Parsed call_context from participant_metadata:", incomingContext);
    } catch {
      console.warn("[TOKEN] Failed to parse participant_metadata as JSON");
    }
  }

  // Also check request.call_context for backward compatibility
  const callContext: CallContext = {
    ...defaultCallContext,
    ...incomingContext,
    ...(request.call_context ?? {}),
  };

  const jwtMetadata = JSON.stringify({
    is_remote: false,
    ...callContext,
  });

  console.log("[TOKEN] JWT metadata to embed:", jwtMetadata);

  const at = new AccessToken(
    process.env.LIVEKIT_API_KEY,
    process.env.LIVEKIT_API_SECRET,
    {
      identity: request.participant_identity,
      ttl: "10m",
      // ✅ Metadata is set ONCE here with call_context embedded.
      // It is NEVER overwritten below.
      metadata: jwtMetadata,
    },
  );

  at.addGrant({
    roomJoin: true,
    room: request.room_name,
    canUpdateOwnMetadata: true,
  });

  if (request.participant_name) {
    at.name = request.participant_name;
  }
  if (request.participant_identity) {
    at.identity = request.participant_identity;
  }
  // ⚠️ Do NOT touch at.metadata here — it's already set above with call_context.
  if (request.participant_attributes) {
    at.attributes = request.participant_attributes;
  }
  if (request.room_config) {
    at.roomConfig = RoomConfiguration.fromJson(request.room_config);
  }

  return at.toJwt();
}

export default async function handleToken(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    res.status(405).end("Method Not Allowed");
    return;
  }
  if (!apiKey || !apiSecret) {
    res.statusMessage = "Environment variables aren't set up correctly";
    res.status(500).end();
    return;
  }

  const options = req.body ?? {};
  console.log(req.body);
  const suffix = crypto.randomUUID().substring(0, 8);
  options.room_name = options.room_name ?? options.roomName ?? `room-${suffix}`;
  options.participant_identity =
    options.participant_identity ?? options.participantName ?? `user-${suffix}`;

  try {
    res.status(200).json({
      server_url: process.env.NEXT_PUBLIC_LIVEKIT_URL,
      participant_token: await createToken(options),
    });
  } catch (err) {
    console.error("Error generating token:", err);
    res.status(500).send({ message: "Generating token failed" });
  }
}
