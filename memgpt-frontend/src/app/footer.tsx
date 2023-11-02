import React from 'react';
import { cnMuted } from '@memgpt/components/typography';

function Footer() {
  const year = new Date().getFullYear();
  return <p className={cnMuted('text-center')}>&copy; {year} MemGPT</p>;
}

export default Footer;
