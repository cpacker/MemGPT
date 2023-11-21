import React from 'react';
import Header from './shared/layout/header';
import Footer from './shared/layout/footer';
import { createBrowserRouter, Outlet, redirect } from 'react-router-dom';
import Chat from './modules/chat/chat';
import { settingsRoute } from './modules/settings/settings.routes';
import Index from './index';
import { chatRoute } from './modules/chat/chat.routes';

const rootRoute = () => <>
  <Header />
  <div className='h-full'>
    <Outlet />
  </div>
  <Footer />
</>;

export const router = createBrowserRouter([
  {
    path: '/',
    element: rootRoute(),
    children: [
      {
        path: '',
        element: <Index/>
      },
      chatRoute,
      settingsRoute
    ]
  },

]);
