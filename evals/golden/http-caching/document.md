# HTTP caching

A cache keeps a copy of a response so the next request for the same resource can be answered without asking the origin server again. Browsers keep a private cache for one user; CDNs and proxies are shared caches that serve many users. Caching is controlled almost entirely by response headers.

## Freshness

A stored response is either fresh or stale. While it is fresh, a cache may serve it without contacting the server at all. The `Cache-Control: max-age=600` directive says the response stays fresh for 600 seconds after it was generated. The `s-maxage` directive does the same but applies only to shared caches such as a CDN, and overrides `max-age` there. A cache reports how long it has held a response in the `Age` header.

## Validation

When a response goes stale, the cache does not have to download it again. It can revalidate: send a conditional request and ask whether its copy is still good.

- With an `ETag` (an opaque version identifier, such as a hash of the content), the cache sends `If-None-Match` with the stored tag.
- With a `Last-Modified` date, the cache sends `If-Modified-Since`.

If nothing changed, the server answers `304 Not Modified` with no body, and the cache keeps using its copy. Only the headers cross the network.

## Directives that are easy to confuse

`no-cache` does not mean "do not cache". It means a cache may store the response but must revalidate it with the server before every use. The directive that forbids storing a response at all is `no-store`, used for sensitive data such as bank statements.

`private` means only the user's browser may store the response; shared caches must not. `public` allows shared caches to store it even when the request carried credentials.

## Cache keys and Vary

A cache looks responses up by a key, normally the method and the URL. The `Vary` header adds request headers to that key. With `Vary: Accept-Encoding`, a gzip-compressed copy and an uncompressed copy are stored as separate variants, so a client that cannot decompress never receives gzip.

## Cache busting

Long lifetimes and fast updates seem to conflict. The usual answer is fingerprinted file names: the build puts a hash of the content in the name, such as `app.3f9a1c.js`. The file can then be served with `max-age=31536000, immutable`, because any change produces a new name and therefore a new URL. The HTML page that references it is kept short-lived.

## About these notes

Written for the doc-to-deck golden set. Last revised November 2026. Corrections welcome.
