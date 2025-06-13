import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';

export default function NoteView() {
  const { id } = useParams();
  const [map, setMap] = useState(null);
  const [content, setContent] = useState('');

  useEffect(() => {
    fetch('/notes.json').then(res => res.json()).then(setMap);
  }, []);

  useEffect(() => {
    if (!map) return;
    const file = map[id];
    if (file) {
      fetch('/' + file).then(res => res.text()).then(setContent);
    }
  }, [map, id]);

  if (!map) return <div>Loading&hellip;</div>;
  if (!map[id]) return <div>Note not found</div>;

  return (
    <div>
      <Link to="/">&larr; Back to graph</Link>
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}
