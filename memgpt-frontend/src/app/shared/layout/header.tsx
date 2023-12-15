import React from 'react';
import { Button } from '@memgpt/components/button';
import { Avatar, AvatarFallback, AvatarImage } from '@memgpt/components/avatar';
import { NavLink } from 'react-router-dom';

const twNavLink = '[&.active]:opacity-100 opacity-60';
const Header = () => (
  <div className='border-b sm:px-8 py-2 flex justify-between items-start'>
    <NavLink to='/'>
      <span className='sr-only'>Home</span>
      <Avatar className='border'>
        <AvatarImage alt='MemGPT logo.' src='/memgpt_logo_transparent.png' />
        <AvatarFallback className='border'>MG</AvatarFallback>
      </Avatar>
    </NavLink>

    <nav className='flex space-x-4'>
      <Button size='sm' asChild variant='link'>
        <NavLink className={twNavLink} to='/'>Home</NavLink>
      </Button>
      <Button size='sm' asChild variant='link'>
        <NavLink className={twNavLink} to='/chat'>Chat</NavLink>
      </Button>
      <Button size='sm' asChild variant='link'>
        {/* @ts-ignore */}
        <NavLink className={twNavLink} to='/settings/agents'>Settings</NavLink>
      </Button>
    </nav>
  </div>
);

export default Header;
