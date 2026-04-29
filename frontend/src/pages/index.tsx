import { AnimatePresence, motion } from "framer-motion";
import { Inter } from "next/font/google";
import Head from "next/head";
import { useState, useCallback } from "react";

import { PlaygroundConnect } from "@/components/PlaygroundConnect";
import Playground from "@/components/playground/Playground";
import { PlaygroundToast, ToastType } from "@/components/toast/PlaygroundToast";
import { ConfigProvider, useConfig } from "@/hooks/useConfig";
import { ToastProvider, useToast } from "@/components/toast/ToasterProvider";
import { TokenSourceConfigurable, TokenSource } from "livekit-client";

// Default call context — sent to /api/token and embedded in the JWT metadata
// so the LiveKit agent can read it from participant metadata on room join.
const CALL_CONTEXT_STORAGE_KEY = "call_context";

export type CallContext = {
  prompt: string;
  provider: string;
  model: string;
  stt_model: string;
  language: string;
  voice: string;
};

const DEFAULT_CALL_CONTEXT: CallContext = {
  prompt: "You are a helpful assistant.",
  provider: "groq",
  model: "llama-3.3-70b-versatile",
  stt_model: "nova-2-general",
  language: "hi",
  voice: "cgSgspJ2msm6clMCkdW9", // Jessica
};

/** Load persisted call context from localStorage, falling back to defaults. */
function loadCallContext(): CallContext {
  if (typeof window === "undefined") return DEFAULT_CALL_CONTEXT;
  try {
    const raw = localStorage.getItem(CALL_CONTEXT_STORAGE_KEY);
    if (raw) return { ...DEFAULT_CALL_CONTEXT, ...JSON.parse(raw) };
  } catch {
    // ignore malformed JSON
  }
  return DEFAULT_CALL_CONTEXT;
}

/** Persist call context to localStorage. */
function saveCallContext(ctx: CallContext) {
  try {
    localStorage.setItem(CALL_CONTEXT_STORAGE_KEY, JSON.stringify(ctx));
  } catch {
    // ignore storage errors (private browsing, quota, etc.)
  }
}

/** Build a TokenSource endpoint that includes the given call_context. */
function makeTokenSource(ctx: CallContext) {
  return TokenSource.endpoint("/api/token", { call_context: ctx });
}

const themeColors = [
  "orange",
  "green",
  "amber",
  "blue",
  "violet",
  "rose",
  "pink",
  "teal",
];

const inter = Inter({ subsets: ["latin"] });

export default function Home() {
  return (
    <ToastProvider>
      <ConfigProvider>
        <HomeInner />
      </ConfigProvider>
    </ToastProvider>
  );
}

export function HomeInner() {
  const { config } = useConfig();
  const { toastMessage, setToastMessage } = useToast();
  const [autoConnect, setAutoConnect] = useState(false);

  // Initialise from localStorage so settings survive page refreshes.
  const [callContext, setCallContextState] = useState<CallContext>(loadCallContext);

  const [tokenSource, setTokenSource] = useState<
    TokenSourceConfigurable | undefined
  >(() => {
    if (process.env.NEXT_PUBLIC_LIVEKIT_URL) {
      return makeTokenSource(loadCallContext());
    }
    return undefined;
  });

  /**
   * Update call context: persists to localStorage and refreshes the token
   * source so the next connection uses the new values.
   */
  const updateCallContext = useCallback(
    (partial: Partial<CallContext>) => {
      setCallContextState((prev) => {
        const next = { ...prev, ...partial };
        saveCallContext(next);
        if (process.env.NEXT_PUBLIC_LIVEKIT_URL) {
          setTokenSource(makeTokenSource(next));
        }
        return next;
      });
    },
    []
  );

  return (
    <>
      <Head>
        <title>{config.title}</title>
        <meta name="description" content={config.description} />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"
        />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black" />
        <meta
          property="og:image"
          content="https://livekit.io/images/og/agents-playground.png"
        />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="relative flex flex-col justify-center px-4 items-center h-full w-full bg-white repeating-square-background">
        <AnimatePresence>
          {toastMessage && (
            <motion.div
              className="left-0 right-0 top-0 absolute z-10"
              initial={{ opacity: 0, translateY: -50 }}
              animate={{ opacity: 1, translateY: 0 }}
              exit={{ opacity: 0, translateY: -50 }}
            >
              <PlaygroundToast />
            </motion.div>
          )}
        </AnimatePresence>
        {tokenSource ? (
          <Playground
            themeColors={themeColors}
            tokenSource={tokenSource}
            autoConnect={autoConnect}
            agentOptions={
              config.settings.agent
                ? { agentName: config.settings.agent }
                : config.agent_dispatch
            }
          />
        ) : (
          <PlaygroundConnect
            accentColor={themeColors[0]}
            onConnectClicked={(tokenSource, shouldAutoConnect) => {
              setTokenSource(tokenSource);
              if (shouldAutoConnect) {
                setAutoConnect(true);
              }
            }}
          />
        )}
      </main>
    </>
  );
}
