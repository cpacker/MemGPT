import React from 'react';
import { Outlet, RootRoute, Router, Route } from '@tanstack/react-router';
import Chat from './modules/chat/chat';
import { settingsRoute } from './modules/settings/settings.routes';
import Header from './shared/layout/header';
import Footer from './shared/layout/footer';

const TanStackRouterDevtools =
  process.env.NODE_ENV === 'production'
    ? () => null // Render nothing in production
    : React.lazy(() =>
      // Lazy load in development
      import('@tanstack/router-devtools').then((res) => ({
        default: res.TanStackRouterDevtools,
        // For Embedded Mode
        // default: res.TanStackRouterDevtoolsPanel
      })),
    );

export const rootRoute = new RootRoute({
  component: () => (<>
      <Header />
      <div className="h-full">
        <Outlet />
      </div>
      <TanStackRouterDevtools initialIsOpen={false} />
      <Footer />
    </>
  ),
});

const indexRoute = new Route({ getParentRoute: () => rootRoute, path: '/', component: Chat });

const routeTree = rootRoute.addChildren([
  indexRoute,
  settingsRoute,
]);

export const router = new Router({
  routeTree: routeTree,
});

declare module '@tanstack/react-router' {
  interface Register {
    // This infers the type of our router and registers it across your entire project
    router: typeof router;
  }
}
