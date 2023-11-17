import { Analytics } from "@vercel/analytics/react";

import { Home } from "./components/home";

export default async function App() {
  return (
    <>
      <Home />
      <Analytics />
    </>
  );
}
