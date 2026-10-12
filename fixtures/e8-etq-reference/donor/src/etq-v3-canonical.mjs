// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Trent Slade / QSOL-IMC.

import { createHash } from "node:crypto";

function assertCanonicalNumber(value) {
  if (!Number.isSafeInteger(value)) {
    throw new TypeError("canonical ETQ JSON permits safe integers only");
  }
}

/** Recursive lexicographic canonical JSON with an integer-only number ABI. */
export function canonicalSerialize(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    assertCanonicalNumber(value);
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalSerialize).join(",")}]`;
  }
  if (typeof value === "object") {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError("canonical ETQ JSON permits plain objects only");
    }
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalSerialize(value[key])}`)
      .join(",")}}`;
  }
  throw new TypeError(`unsupported canonical ETQ value type: ${typeof value}`);
}

export function utf8(value) {
  return Buffer.from(value, "utf8");
}

export function sha256Bytes(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export function domainSeparatedSha256(domain, bytes) {
  if (typeof domain !== "string" || domain.length === 0 || domain.includes("\0")) {
    throw new TypeError("hash domain must be a non-empty NUL-free string");
  }
  return createHash("sha256")
    .update(domain, "utf8")
    .update(Buffer.from([0]))
    .update(bytes)
    .digest("hex");
}

export function canonicalObjectSha256(domain, value) {
  return domainSeparatedSha256(domain, utf8(canonicalSerialize(value)));
}
