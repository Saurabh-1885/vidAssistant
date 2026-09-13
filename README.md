# Scribe — AI Video Assistant (Frontend UI)

Frosted-glass workspace UI for an AI meeting/video assistant: video or URL input,
English/Hinglish toggle, summary + searchable transcript, action items, key
decisions, open questions, and a persistent meeting chat.

## What's in the bundle

```
src/
  routes/index.tsx          # the whole app UI (session card, panels, chat)
  routes/__root.tsx         # root layout, fonts, metadata
  styles.css                # Tailwind v4 theme + glass/reveal animation utilities
  components/ai-elements/   # Conversation, Message, PromptInput, Shimmer
  components/ui/            # shadcn-style primitives used by the above
  lib/utils.ts              # cn() helper
```

## Requirements

- React 19 + Vite
- Tailwind CSS v4 (the `styles.css` here uses the `@theme` syntax — plain
  Tailwind 3 configs won't pick up the tokens)
- TypeScript

Key npm dependencies (see `package.json` for exact versions):

```
@tanstack/react-router @tanstack/react-start @tanstack/react-query
@radix-ui/react-tooltip @radix-ui/react-dropdown-menu @radix-ui/react-select
@radix-ui/react-hover-card @radix-ui/react-dialog @radix-ui/react-slot
@radix-ui/react-scroll-area class-variance-authority clsx tailwind-merge
lucide-react motion
```

If you use a plain Vite SPA instead of TanStack Start, keep `index.tsx` as a
page component (`export default function Index()`) and copy the `<head>` font
links from `__root.tsx` into your `index.html`.

## Connecting your Python pipeline

`src/routes/index.tsx` currently simulates everything so the UI works standalone.
Two functions to replace:

1. **`analyze()`** (~middle of the file, 1500 ms timeout) — replace the simulated
   result with a call to your FastAPI backend, e.g.:

```ts
const res = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ source, language }),
});
const data = await res.json();
// expects the shape returned by run_pipeline():
// { summary, transcript: [{speaker, text}], action_items, key_decisions, questions }
```

2. **`answerFor(text)`** (chat reply) — replace keyword routing with a call to
   `ask_question` on your backend:

```ts
const res = await fetch("http://localhost:8000/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: text }),
});
return (await res.json()).answer;
```

Add CORS on the FastAPI side (`allow_origins=["http://localhost:5173"]`) so the
browser can reach it.

## Notes

- Meeting chat persists in browser `localStorage` under the key
  `scribe-meeting-chat` (one conversation, as configured).
- Animations: `reveal-on-scroll` (fade-in on scroll via IntersectionObserver),
  `animate-soft-enter` (entrance), hover transitions on cards/buttons. All are
  disabled automatically for users with `prefers-reduced-motion` enabled.
- Fonts (DM Sans + Space Grotesk) load via Google Fonts links in `__root.tsx`.
