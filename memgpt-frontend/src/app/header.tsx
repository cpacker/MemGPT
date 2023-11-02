import React from 'react';
import { cnH1, cnLead } from '@memgpt/components/typography';

const Header = () => (
  <>
    <h1 className={cnH1()}>MemGPT</h1>
    <p className={cnLead()}>Towards LLMs as Operating Systems</p>
  </>
);

export default Header;
