import Footer from './footer';
import UserInput from './user-input';
import Header from './header';
import MessageContainer from './messages/message-container';
import { useCallback, useEffect, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import { useMessages } from './use-messages';
import StatusIndicator from './status-indicator';


export function App() {
  const { readyState, messageHistory, sendMessage } = useMessages();

  return (
    <div className='mx-auto max-w-screen-lg p-4'>
      <Header>
        <StatusIndicator readyState={readyState}/>
      </Header>
        <MessageContainer messages={messageHistory} />
      <UserInput onSend={sendMessage} />
      <Footer />
    </div>
  );
}

export default App;
