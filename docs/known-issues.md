# Known issues

Problems that are understood, not yet fixed, and worth knowing about before you
rely on PrivateAIStack. Each entry says what happens, who it affects, and what
to do in the meantime. Fixed issues move to the [changelog](../CHANGELOG.md).

## The default database URL contains a built-in password

**Status:** open. Present in every published version through 0.2.0b3.

**What happens.** `database_url` defaults to a PostgreSQL DSN whose username
and password are both `private_ai_stack`. That is the credential the bundled
development Compose file creates. Production mode (`environment=production`)
requires an API key, but does not refuse this default DSN.

**Who is affected.** A deployment that sets `environment=production` and
leaves `DATABASE_URL` unset, or reuses the development password for a database
other processes can reach. The bundled Compose stack is documented as
development-oriented.

**What to do now.** Always set `DATABASE_URL` yourself outside development,
with a password of your own, and do not expose the database port beyond the
stack.

**Planned fix.** Refuse the built-in DSN when `environment=production`.

## Request validation errors echo the submitted input

**Status:** open. Present in every published version through 0.2.0b3.

**What happens.** When a request body fails validation, the API answers `422`
with a `details.errors` list taken directly from the validator. Each entry
includes the `input` that was rejected, so whatever the client sent in that
field is returned in the response.

**Who is affected.** The value goes back only to the client that sent it, so
this discloses nothing to a third party by itself. It matters if responses are
logged by a proxy or gateway, or shown to someone other than the caller: a
secret mistakenly placed in a request field would then be recorded there.

**What to do now.** Do not log `422` response bodies from this API verbatim,
or strip `details.errors[].input` before storing them.

**Planned fix.** Drop `input` from returned validation errors and keep the
field path and the reason.
