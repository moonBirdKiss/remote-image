FROM alpine:3.20

WORKDIR /app

RUN apk add --no-cache ca-certificates \
    && adduser -D -H -u 10001 appuser
