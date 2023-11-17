"use client";

import React, { useContext, useEffect, useState } from "react";

import { QueryClient, QueryClientProvider } from "react-query";
import { useMobileScreen } from "../utils/mobile";

import dynamic from "next/dynamic";
import { Path } from "../constant";
import { ErrorBoundary } from "./layout/error";

import {
  Route,
  HashRouter as Router,
  Routes,
  useNavigate,
} from "react-router-dom";
import { Bot, useBotStore } from "../store/bot";
import { SideBar } from "./layout/sidebar";
import { LoadingPage } from "@/app/components/ui/loading";

const SettingsPage = dynamic(
  async () => (await import("./settings")).Settings,
  {
    loading: () => <LoadingPage />,
  },
);

const ChatPage = dynamic(async () => (await import("./chat/chat")).Chat, {
  loading: () => <LoadingPage />,
});

const useHasHydrated = () => {
  const [hasHydrated, setHasHydrated] = useState<boolean>(false);

  useEffect(() => {
    setHasHydrated(true);
  }, []);

  return hasHydrated;
};

const loadAsyncGoogleFont = () => {
  const linkEl = document.createElement("link");
  const googleFontUrl = "https://fonts.googleapis.com";
  linkEl.rel = "stylesheet";
  linkEl.href =
    googleFontUrl + "/css2?family=Noto+Sans:wght@300;400;700;900&display=swap";
  document.head.appendChild(linkEl);
};

// if a bot is passed this HOC ensures that the bot is added to the store
// and that the user can directly have a chat session with it
function withBot(Component: React.FunctionComponent, bot?: Bot) {
  return function WithBotComponent() {
    const [botInitialized, setBotInitialized] = useState(false);
    const navigate = useNavigate();
    const botStore = useBotStore();
    if (bot && !botInitialized) {
      if (!bot.share?.id) {
        throw new Error("bot must have a shared id");
      }
      // ensure that bot for the same share id is not created a 2nd time
      let sharedBot = botStore.getByShareId(bot.share?.id);
      if (!sharedBot) {
        sharedBot = botStore.create(bot, { readOnly: true });
      }
      // let the user directly chat with the bot
      botStore.selectBot(sharedBot.id);
      setTimeout(() => {
        // redirect to chat - use history API to clear URL
        history.pushState({}, "", "/");
        navigate(Path.Chat);
      }, 1);
      setBotInitialized(true);
      return <LoadingPage />;
    }

    return <Component />;
  };
}

const SidebarContext = React.createContext<{
  showSidebar: boolean;
  setShowSidebar: (show: boolean) => void;
} | null>(null);

function SidebarContextProvider(props: { children: React.ReactNode }) {
  const [showSidebar, setShowSidebar] = useState(true);
  return (
    <SidebarContext.Provider value={{ showSidebar, setShowSidebar }}>
      {props.children}
    </SidebarContext.Provider>
  );
}

export const useSidebarContext = () => {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error(
      "useSidebarContext must be used within an SidebarContextProvider",
    );
  }
  return context;
};

function Screen() {
  const isMobileScreen = useMobileScreen();
  const { showSidebar } = useSidebarContext();

  const showSidebarOnMobile = showSidebar || !isMobileScreen;

  useEffect(() => {
    loadAsyncGoogleFont();
  }, []);

  return (
    <main className="flex overflow-hidden h-screen w-screen box-border">
      <>
        {showSidebarOnMobile && <SideBar />}
        <div className="flex-1 overflow-hidden">
          <Routes>
            <Route path={Path.Chat} element={<ChatPage />} />
            <Route path={Path.Settings} element={<SettingsPage />} />
          </Routes>
        </div>
      </>
    </main>
  );
}

export function Home({ bot }: { bot?: Bot }) {
  if (!useHasHydrated()) {
    return <LoadingPage />;
  }

  const BotScreen = withBot(Screen, bot);
  const queryClient = new QueryClient();

  return (
    <ErrorBoundary>
      <Router>
        <QueryClientProvider client={queryClient}>
          <SidebarContextProvider>
            <BotScreen />
          </SidebarContextProvider>
        </QueryClientProvider>
      </Router>
    </ErrorBoundary>
  );
}
