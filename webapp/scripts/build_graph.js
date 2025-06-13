const fs = require('fs');
const path = require('path');

const vaultDir = path.resolve(__dirname, '..', 'StateFarm Interview');
const publicDir = path.resolve(__dirname, '..', 'webapp', 'public');
const notesDir = path.join(publicDir, 'notes');

function walk(dir, callback) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, callback);
    } else if (entry.isFile()) {
      callback(full);
    }
  }
}

const nodes = [];
const links = [];
const map = {};

walk(vaultDir, file => {
  if (!file.endsWith('.md')) return;
  const rel = path.relative(vaultDir, file);
  const name = rel.replace(/\.md$/, '');
  const content = fs.readFileSync(file, 'utf-8');
  nodes.push({ id: name });
  map[name] = `notes/${encodeURIComponent(rel)}`;

  const regex = /\[\[([^\]]+)\]\]/g;
  let m;
  while ((m = regex.exec(content))) {
    links.push({ source: name, target: m[1] });
  }

  const dest = path.join(notesDir, rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(file, dest);
});

fs.mkdirSync(publicDir, { recursive: true });
fs.writeFileSync(path.join(publicDir, 'graph.json'), JSON.stringify({ nodes, links }, null, 2));
fs.writeFileSync(path.join(publicDir, 'notes.json'), JSON.stringify(map, null, 2));
console.log(`Generated ${nodes.length} notes`);
