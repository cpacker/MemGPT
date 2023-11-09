import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import React from 'react';
import { RouterProvider } from '@tanstack/react-router';
import { router } from './router';
import { Toaster } from '@memgpt/components/toast';

const queryClient = new QueryClient();

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <ReactQueryDevtools initialIsOpen={false} />
      <Toaster/>
    </QueryClientProvider>
  );
}

export default App;
