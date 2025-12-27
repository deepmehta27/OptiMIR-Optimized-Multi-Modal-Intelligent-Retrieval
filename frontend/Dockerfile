FROM node:20-alpine AS builder
WORKDIR /app

# FIX: Copy directly from the current context
COPY package*.json ./
RUN npm ci

# FIX: Copy all frontend source code from current dir
COPY . ./
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app ./
EXPOSE 3000
CMD ["npm", "start"]