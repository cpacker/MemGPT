import React from 'react';
import { Link } from '@tanstack/react-router';
import { Button } from '@memgpt/components/button';
import { Avatar, AvatarFallback, AvatarImage } from '@memgpt/components/avatar';

const twLink ="data-[status=active]:opacity-100 opacity-60";
const Header = () => (
  <div className="border-b sm:px-8 py-2 flex justify-between items-start">
    <Link to="/">
      <span className="sr-only">Home</span>
      <Avatar className="border">
        <AvatarImage alt="MemGPT logo." src="/memgpt_logo_transparent.png" />
        <AvatarFallback className="border">MG</AvatarFallback>
      </Avatar>
    </Link>

    <nav className="flex space-x-4">
      <Button size="sm" asChild variant="link">
        <Link className={twLink} to="/">Chat</Link>
      </Button>
      <Button size="sm" asChild variant="link">
        {/* @ts-ignore */}
        <Link className={twLink} to="/settings/agents">Settings</Link>
      </Button>
    </nav>
  </div>
);

export default Header;
