import Footer from './footer';
import UserInput from './user-input';
import Header from './header';
import MessageContainer from './messages/message-container';
import { useCallback, useEffect, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
export function App() {
  const [socketUrl, setSocketUrl] = useState('ws://localhost:8000/api/ws');
  const [messageHistory, setMessageHistory] = useState<string[]>([]);

  const { sendMessage, lastMessage, readyState } = useWebSocket<string>(socketUrl);

  useEffect(() => {
    if (lastMessage !== null) {
      setMessageHistory((prev) => [...prev, lastMessage.data]);
    }
  }, [lastMessage, setMessageHistory]);

  const handleClickSendMessage = useCallback((message: string) => {
    console.log('calling');
    setMessageHistory((h) => [...h, message]);
    sendMessage(message);
  }, []);

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Open',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  return (
    <div className="mx-auto max-w-screen-lg p-4">
      <Header />
      <MessageContainer messages={messageHistory} />
      <UserInput onSend={handleClickSendMessage} />
      <Footer />
    </div>
  );
}

export default App;
