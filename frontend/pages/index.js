import Head from "next/head";
import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [advice, setAdvice] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setResults([]);
    setAdvice("");

    try {
      // Search for matching outfits
      const searchRes = await fetch(`${API_URL}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
      });
      const searchData = await searchRes.json();

      if (!searchRes.ok) {
        throw new Error(searchData.error || "Search failed");
      }

      setResults(searchData.results);

      // Get AI advice
      const adviceRes = await fetch(`${API_URL}/api/advice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          results: searchData.results,
        }),
      });
      const adviceData = await adviceRes.json();
      setAdvice(adviceData.advice);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Head>
        <title>Fashion Advisor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div style={styles.container}>
        <h1 style={styles.title}>Fashion Advisor</h1>
        <p style={styles.subtitle}>
          Describe what you&apos;re looking for and get outfit recommendations
        </p>

        <form onSubmit={handleSearch} style={styles.form}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. red floral summer dress, black leather jacket..."
            style={styles.input}
            disabled={loading}
          />
          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        {error && <p style={styles.error}>{error}</p>}

        {advice && (
          <div style={styles.adviceBox}>
            <strong>AI Fashion Advice:</strong>
            <p style={{ margin: "8px 0 0" }}>{advice}</p>
          </div>
        )}

        {results.length > 0 && (
          <div style={styles.grid}>
            {results.map((item, i) => (
              <div key={item.id} style={styles.card}>
                <div style={styles.rank}>#{i + 1}</div>
                <img
                  src={`${API_URL}${item.image_url}`}
                  alt={item.description}
                  style={styles.image}
                />
                <div style={styles.cardBody}>
                  <div style={styles.score}>
                    Score: {(item.score * 100).toFixed(1)}%
                  </div>
                  <div style={styles.categories}>
                    {item.categories.map((cat) => (
                      <span key={cat} style={styles.tag}>
                        {cat}
                      </span>
                    ))}
                  </div>
                  {item.colors && item.colors.length > 0 && (
                    <div style={styles.colors}>
                      {item.colors.map((color) => (
                        <span key={color} style={styles.colorTag}>
                          {color}
                        </span>
                      ))}
                    </div>
                  )}
                  <div style={styles.attributes}>
                    {item.attributes.slice(0, 5).map((attr) => (
                      <span key={attr} style={styles.attrTag}>
                        {attr}
                      </span>
                    ))}
                    {item.attributes.length > 5 && (
                      <span style={styles.attrTag}>
                        +{item.attributes.length - 5} more
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

const styles = {
  container: {
    maxWidth: 900,
    margin: "0 auto",
    padding: "40px 20px",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  title: {
    fontSize: 32,
    fontWeight: 700,
    textAlign: "center",
    margin: 0,
  },
  subtitle: {
    textAlign: "center",
    color: "#666",
    marginBottom: 30,
  },
  form: {
    display: "flex",
    gap: 10,
    marginBottom: 24,
  },
  input: {
    flex: 1,
    padding: "12px 16px",
    fontSize: 16,
    border: "2px solid #ddd",
    borderRadius: 8,
    outline: "none",
  },
  button: {
    padding: "12px 24px",
    fontSize: 16,
    fontWeight: 600,
    background: "#000",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
  },
  error: {
    color: "red",
    textAlign: "center",
  },
  adviceBox: {
    background: "#f0f7ff",
    border: "1px solid #c0d8f0",
    borderRadius: 8,
    padding: 16,
    marginBottom: 24,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))",
    gap: 16,
  },
  card: {
    border: "1px solid #eee",
    borderRadius: 8,
    overflow: "hidden",
    position: "relative",
    background: "#fff",
    boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
  },
  rank: {
    position: "absolute",
    top: 8,
    left: 8,
    background: "#000",
    color: "#fff",
    borderRadius: 6,
    padding: "4px 10px",
    fontSize: 14,
    fontWeight: 700,
  },
  image: {
    width: "100%",
    height: 280,
    objectFit: "cover",
  },
  cardBody: {
    padding: 12,
  },
  score: {
    fontWeight: 700,
    fontSize: 15,
    marginBottom: 8,
  },
  categories: {
    display: "flex",
    flexWrap: "wrap",
    gap: 4,
    marginBottom: 6,
  },
  tag: {
    background: "#000",
    color: "#fff",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 12,
  },
  colors: {
    display: "flex",
    flexWrap: "wrap",
    gap: 4,
    marginBottom: 6,
  },
  colorTag: {
    background: "#e8f4e8",
    color: "#2d6a2d",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 12,
    fontWeight: 600,
  },
  attributes: {
    display: "flex",
    flexWrap: "wrap",
    gap: 4,
  },
  attrTag: {
    background: "#f0f0f0",
    color: "#555",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 11,
  },
};
