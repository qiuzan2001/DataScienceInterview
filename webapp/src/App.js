import React from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import GraphView from './components/GraphView';
import NoteView from './components/NoteView';

export default function App() {
  return (
    <div>
      <header>
        <h1>Obsidian Notes</h1>
        <nav>
          <Link to="/">Graph</Link>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<GraphView />} />
        <Route path="/note/:id" element={<NoteView />} />
      </Routes>
    </div>
  );
}
