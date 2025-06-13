# Publishing Obsidian Notes as a Website

This repository contains a folder called `StateFarm Interview` which is an Obsidian vault. If you want to publish these notes online with a graph view similar to Obsidian's own Graph, you can use the **Obsidian Digital Garden** plugin. It lets you push your notes to a GitHub repository and serves them through Vercel with interactive navigation, backlinks and a graph.

## Steps

1. **Create accounts**
   - Sign up for a free GitHub account if you do not already have one.
   - Sign up for a free account on Vercel (can authenticate with GitHub).

2. **Deploy the Digital Garden template**
   - Open <https://github.com/oleeskild/digitalgarden> in your browser.
   - Click the `Deploy to Vercel` button and follow the prompts. This copies the template to your GitHub account and deploys the site to Vercel. You will get a URL like `https://your-garden.vercel.app`.

3. **Generate a GitHub access token**
   - In GitHub, create a *fine‑grained* Personal Access Token with "Contents" and "Pull requests" set to `Read and write` for the repository created in the previous step. Save this token—it will be used by the plugin to publish notes.

4. **Install the Obsidian Digital Garden plugin**
   - In Obsidian, open **Settings → Community Plugins** and search for `Obsidian Digital Garden`. Install and enable it.

5. **Configure the plugin**
   - Open the plugin settings and fill in:
       - **GitHub Username** – your GitHub username.
       - **GitHub Repo Name** – the repo created from the template (`digitalgarden` or whatever name you chose).
       - **GitHub Token** – the token created in step 3.
       - **Base URL** – the Vercel site URL.
   - Adjust additional options as desired (for example, enable the graph view).

6. **Publish your notes**
   - In Obsidian, open the command palette and run `Digital Garden: Publish Single Note` on notes you want to make public. You can also publish multiple notes or all notes using the plugin commands.
   - After the site builds (Vercel automatically does this), your published notes will be available online with a graph view, backlinks and search. Links between notes behave the same way as in Obsidian.

7. **Update or customize the site**
   - The static website lives in the GitHub repository you created. You can edit the code there to change styling or add features. When you push changes, Vercel redeploys the site.

## Alternatives

If you prefer a simpler static site without the interactive graph, you can explore tools like [Quartz](https://github.com/jackyzha0/quartz) or [Obsidian-Mkdocs](https://github.com/jobindjohn/obsidian-mkdocs). These convert your vault into a static website. However, the **Obsidian Digital Garden** plugin offers the closest experience to browsing notes directly in Obsidian, including a graph view.

## Local React demo

For a self-hosted approach, the repository also includes a minimal React project
in the `webapp` folder. It renders Markdown files and visualizes their links as
a graph. After installing dependencies, run the `generate` script to copy notes
and build `graph.json`:

```bash
cd webapp
npm install          # install React dependencies
npm run generate     # copy notes and create graph.json
npm start            # launch the development server
```

The site will be available at <http://localhost:3000>. You can navigate through
notes and click nodes in the graph to jump between them.
