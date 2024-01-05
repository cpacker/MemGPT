import React from 'react';
import Header from './shared/layout/header';
import Footer from './shared/layout/footer';
import { createBrowserRouter, Outlet } from 'react-router-dom';
import { settingsRoute } from './modules/settings/settings.routes';
import Home from './modules/home/home';
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
        element: <Home/>
      },
      chatRoute,
      settingsRoute
    ]
  },

]);
