/**
 * tools.ts — Production-ready tool definitions for the customer support agent.
 *
 * Each tool uses Zod for schema validation, returns error objects (never throws),
 * and includes realistic mock implementations with logging.
 */

import { tool } from "ai";
import { z } from "zod";

// ──── Internal helpers ────────────────────────────────────────────────────────

function log(toolName: string, input: unknown, output: unknown): void {
  console.log(`[tool:${toolName}]`, JSON.stringify({ input, output }, null, 2));
}

// ──── Mock data ───────────────────────────────────────────────────────────────

const KB_ARTICLES = [
  {
    id: "kb-001",
    category: "returns",
    title: "How to return an item",
    content:
      "Returns are accepted within 30 days of delivery. Items must be unused and in original packaging. Visit returns.acme.com to start your return.",
    url: "https://help.acme.com/returns/how-to",
  },
  {
    id: "kb-002",
    category: "orders",
    title: "Tracking your order",
    content:
      "Once shipped, you'll receive an email with a tracking link. Orders typically ship within 2 business days. Track at track.acme.com.",
    url: "https://help.acme.com/orders/tracking",
  },
  {
    id: "kb-003",
    category: "orders",
    title: "Order not received",
    content:
      "If your tracking shows delivered but you haven't received the package, wait 24 hours then contact us. We'll file a carrier claim and reship or refund.",
    url: "https://help.acme.com/orders/not-received",
  },
  {
    id: "kb-004",
    category: "billing",
    title: "Refund processing times",
    content:
      "Refunds are processed within 3–5 business days after we receive the return. Credit card refunds take an additional 2–3 days to appear.",
    url: "https://help.acme.com/billing/refunds",
  },
  {
    id: "kb-005",
    category: "technical",
    title: "Account login issues",
    content:
      "If you're locked out of your account, use the 'Forgot Password' link. For persistent issues, contact support with your registered email.",
    url: "https://help.acme.com/account/login",
  },
  {
    id: "kb-006",
    category: "general",
    title: "Contact support",
    content:
      "Support is available Monday–Friday, 9AM–6PM ET. Chat, email, or call 1-800-ACME-HELP.",
    url: "https://help.acme.com/contact",
  },
];

const ORDERS_DB: Record<
  string,
  {
    orderNumber: string;
    email: string;
    date: string;
    status: string;
    items: { name: string; qty: number; price: number }[];
    total: number;
    tracking?: string;
    deliveredAt?: string;
  }
> = {
  "ORD-12345": {
    orderNumber: "ORD-12345",
    email: "alice@example.com",
    date: "2026-04-28",
    status: "delivered",
    items: [{ name: "Wireless Headphones", qty: 1, price: 79.99 }],
    total: 79.99,
    tracking: "1Z999AA10123456784",
    deliveredAt: "2026-05-01",
  },
  "ORD-99999": {
    orderNumber: "ORD-99999",
    email: "bob@example.com",
    date: "2026-05-02",
    status: "in_transit",
    items: [
      { name: "Mechanical Keyboard", qty: 1, price: 129.99 },
      { name: "USB-C Hub", qty: 1, price: 39.99 },
    ],
    total: 169.98,
    tracking: "1Z999AA10123456785",
  },
};

const CUSTOMERS_DB: Record<
  string,
  { email: string; name: string; orderNumbers: string[] }
> = {
  "alice@example.com": {
    email: "alice@example.com",
    name: "Alice Chen",
    orderNumbers: ["ORD-12345"],
  },
  "bob@example.com": {
    email: "bob@example.com",
    name: "Bob Smith",
    orderNumbers: ["ORD-99999"],
  },
};

// ──── Tool: searchKnowledgeBase ───────────────────────────────────────────────

/**
 * Searches the support knowledge base for help articles matching the query.
 * Use this before creating a ticket — most issues have a self-service answer.
 */
export const searchKnowledgeBase = tool({
  description:
    "Search the support knowledge base for articles that answer the customer's question. " +
    "Use this FIRST before looking up orders or creating tickets. " +
    "Good for: return policies, shipping timelines, billing questions, account help.",
  parameters: z.object({
    query: z
      .string()
      .min(3)
      .describe(
        "Search query. Be specific: 'how to return an item' not 'returns'."
      ),
    category: z
      .enum(["orders", "returns", "billing", "technical", "general"])
      .optional()
      .describe(
        "Narrow results to a specific category. Omit to search all categories."
      ),
  }),
  execute: async ({ query, category }) => {
    const q = query.toLowerCase();
    const results = KB_ARTICLES.filter((a) => {
      const matchesCategory = !category || a.category === category;
      const matchesQuery =
        a.title.toLowerCase().includes(q) ||
        a.content.toLowerCase().includes(q) ||
        q.split(" ").some((word) => a.content.toLowerCase().includes(word));
      return matchesCategory && matchesQuery;
    })
      .slice(0, 3)
      .map((a) => ({
        id: a.id,
        title: a.title,
        content: a.content,
        category: a.category,
        url: a.url,
      }));

    const output = {
      found: results.length > 0,
      count: results.length,
      articles: results,
    };
    log("searchKnowledgeBase", { query, category }, output);
    return output;
  },
});

// ──── Tool: lookupOrder ───────────────────────────────────────────────────────

/**
 * Looks up a specific order by order number or customer email.
 * Always use this when the customer mentions an order number or asks "where is my order".
 */
export const lookupOrder = tool({
  description:
    "Look up one or more orders by order number or customer email. " +
    "Use when the customer mentions an order number (e.g. 'ORD-12345') or asks about their order status. " +
    "Returns order status, items, tracking number, and delivery date if available.",
  parameters: z.object({
    orderNumber: z
      .string()
      .optional()
      .describe(
        "Order number in format ORD-XXXXX. Provide if the customer gave one."
      ),
    email: z
      .string()
      .email()
      .optional()
      .describe(
        "Customer email to find all their orders. Use if no order number was given."
      ),
  }),
  execute: async ({ orderNumber, email }) => {
    if (!orderNumber && !email) {
      const output = {
        error: "Either orderNumber or email is required.",
        found: false,
      };
      log("lookupOrder", { orderNumber, email }, output);
      return output;
    }

    let orders: (typeof ORDERS_DB)[string][] = [];

    if (orderNumber) {
      const order = ORDERS_DB[orderNumber.toUpperCase()];
      if (order) orders = [order];
    } else if (email) {
      orders = Object.values(ORDERS_DB).filter(
        (o) => o.email.toLowerCase() === email.toLowerCase()
      );
    }

    if (orders.length === 0) {
      const output = {
        found: false,
        message: "No orders found matching the provided information.",
      };
      log("lookupOrder", { orderNumber, email }, output);
      return output;
    }

    const output = {
      found: true,
      orders: orders.map((o) => ({
        orderNumber: o.orderNumber,
        date: o.date,
        status: o.status,
        items: o.items,
        total: o.total,
        tracking: o.tracking ?? null,
        deliveredAt: o.deliveredAt ?? null,
      })),
    };
    log("lookupOrder", { orderNumber, email }, output);
    return output;
  },
});

// ──── Tool: lookupCustomer ────────────────────────────────────────────────────

/**
 * Looks up a customer's profile and full order history by email address.
 * Use this to personalize the conversation and see all orders at once.
 */
export const lookupCustomer = tool({
  description:
    "Look up a customer's profile by email address, including their name and full order history. " +
    "Use when you need to personalize the response or see all orders for a customer at once. " +
    "More complete than lookupOrder when you have the customer's email.",
  parameters: z.object({
    email: z
      .string()
      .email()
      .describe("The customer's email address. Must be a valid email format."),
  }),
  execute: async ({ email }) => {
    const customer = CUSTOMERS_DB[email.toLowerCase()];

    if (!customer) {
      const output = {
        found: false,
        message: `No customer found with email ${email}.`,
      };
      log("lookupCustomer", { email }, output);
      return output;
    }

    const orders = customer.orderNumbers
      .map((num) => ORDERS_DB[num])
      .filter(Boolean)
      .map((o) => ({
        orderNumber: o.orderNumber,
        date: o.date,
        status: o.status,
        total: o.total,
        itemCount: o.items.length,
      }));

    const output = {
      found: true,
      customer: {
        name: customer.name,
        email: customer.email,
        totalOrders: orders.length,
        orders,
      },
    };
    log("lookupCustomer", { email }, output);
    return output;
  },
});

// ──── Tool: createTicket ──────────────────────────────────────────────────────

/**
 * Creates a support ticket for issues that cannot be resolved in the current conversation.
 * Only use this after attempting to resolve with knowledge base and order tools.
 */
export const createTicket = tool({
  description:
    "Create a support ticket for issues that cannot be resolved immediately. " +
    "Only call this AFTER searching the knowledge base and looking up orders. " +
    "Do NOT create a ticket if the knowledge base already answers the question. " +
    "The customer will receive an email confirmation with the ticket ID.",
  parameters: z.object({
    subject: z
      .string()
      .max(100)
      .describe("Brief one-line summary of the issue. Max 100 characters."),
    description: z
      .string()
      .min(20)
      .describe(
        "Full description of the problem including all relevant details."
      ),
    priority: z
      .enum(["low", "medium", "high", "urgent"])
      .describe(
        "Priority level. Use 'urgent' only for fraud or safety issues. 'high' for undelivered orders."
      ),
    customerEmail: z
      .string()
      .email()
      .describe("Customer's email address to send the confirmation to."),
    relatedOrderNumber: z
      .string()
      .optional()
      .describe("Order number associated with the issue, if any."),
    category: z
      .enum(["orders", "returns", "billing", "technical", "general"])
      .describe("Category for routing to the correct support team."),
  }),
  execute: async ({
    subject,
    description,
    priority,
    customerEmail,
    relatedOrderNumber,
    category,
  }) => {
    const ticketId = `TKT-${Date.now().toString(36).toUpperCase()}`;
    const estimatedResponseTimes: Record<string, string> = {
      urgent: "within 2 hours",
      high: "within 4 hours",
      medium: "within 1 business day",
      low: "within 2 business days",
    };

    const output = {
      success: true,
      ticketId,
      subject,
      priority,
      category,
      status: "open",
      estimatedResponseTime: estimatedResponseTimes[priority],
      confirmationSent: true,
      confirmationEmail: customerEmail,
      relatedOrderNumber: relatedOrderNumber ?? null,
    };
    log(
      "createTicket",
      { subject, priority, customerEmail, relatedOrderNumber, category },
      output
    );
    return output;
  },
});

// ──── Tool: escalateToHuman ───────────────────────────────────────────────────

/**
 * Escalates the conversation to a human agent.
 * Use for complex disputes, fraud claims, emotionally distressed customers, or VIP accounts.
 */
export const escalateToHuman = tool({
  description:
    "Escalate this conversation to a live human agent. " +
    "Use when: the customer is very upset, the issue is complex/disputed, " +
    "fraud is suspected, or the customer explicitly requests a human. " +
    "The human agent will see the full conversation history.",
  parameters: z.object({
    reason: z
      .enum([
        "customer_request",
        "complex_dispute",
        "fraud_suspected",
        "vip_customer",
        "repeated_contact",
        "agent_cannot_resolve",
      ])
      .describe("The reason for escalation."),
    urgency: z
      .enum(["normal", "high"])
      .describe(
        "'high' if the customer is very upset or the issue is time-sensitive."
      ),
    summary: z
      .string()
      .min(20)
      .describe(
        "Brief summary of the issue and what has already been attempted. The human agent will read this first."
      ),
    customerEmail: z
      .string()
      .email()
      .describe("Customer's email for the human agent to use."),
  }),
  execute: async ({ reason, urgency, summary, customerEmail }) => {
    const queuePositions: Record<string, number> = {
      normal: Math.floor(Math.random() * 5) + 3,
      high: 1,
    };

    const output = {
      success: true,
      escalated: true,
      reason,
      urgency,
      queuePosition: queuePositions[urgency],
      estimatedWaitMinutes: urgency === "high" ? 2 : 8,
      agentAssigned: false,
      message:
        urgency === "high"
          ? "A senior agent will join within 2 minutes."
          : `You're #${queuePositions.normal} in the queue. An agent will join shortly.`,
      customerEmail,
    };
    log("escalateToHuman", { reason, urgency, customerEmail }, output);
    return output;
  },
});

// ──── Tool: checkReturnEligibility ────────────────────────────────────────────

/**
 * Checks whether an order is eligible for return based on delivery date and policy.
 * Use before helping a customer start a return to set accurate expectations.
 */
export const checkReturnEligibility = tool({
  description:
    "Check if an order is eligible for return based on our 30-day return policy. " +
    "Use this before helping a customer initiate a return so you can set accurate expectations. " +
    "Returns eligibility status, days remaining, and any restrictions.",
  parameters: z.object({
    orderNumber: z
      .string()
      .describe("The order number to check, e.g. 'ORD-12345'."),
    reason: z
      .enum([
        "changed_mind",
        "defective",
        "wrong_item",
        "not_as_described",
        "damaged_in_shipping",
      ])
      .describe("The return reason. Defective/wrong/damaged items may have extended eligibility."),
  }),
  execute: async ({ orderNumber, reason }) => {
    const order = ORDERS_DB[orderNumber.toUpperCase()];

    if (!order) {
      const output = {
        eligible: false,
        error: `Order ${orderNumber} not found.`,
      };
      log("checkReturnEligibility", { orderNumber, reason }, output);
      return output;
    }

    if (order.status !== "delivered") {
      const output = {
        eligible: false,
        reason: `Order is currently '${order.status}' — returns can only be initiated after delivery.`,
        orderStatus: order.status,
      };
      log("checkReturnEligibility", { orderNumber, reason }, output);
      return output;
    }

    const deliveredDate = new Date(order.deliveredAt!);
    const today = new Date("2026-05-06"); // pinned to current date for demo
    const daysSinceDelivery = Math.floor(
      (today.getTime() - deliveredDate.getTime()) / (1000 * 60 * 60 * 24)
    );

    // Extended window for defective/wrong/damaged items
    const extendedReasons = ["defective", "wrong_item", "damaged_in_shipping"];
    const returnWindowDays = extendedReasons.includes(reason) ? 60 : 30;
    const daysRemaining = returnWindowDays - daysSinceDelivery;
    const eligible = daysRemaining > 0;

    const output = {
      eligible,
      orderNumber,
      deliveredAt: order.deliveredAt,
      daysSinceDelivery,
      returnWindowDays,
      daysRemaining: eligible ? daysRemaining : 0,
      reason: eligible
        ? `Eligible for return. ${daysRemaining} day(s) remaining.`
        : `Return window expired ${Math.abs(daysRemaining)} day(s) ago.`,
      returnLabel: eligible ? "https://returns.acme.com/start" : null,
      items: order.items,
    };
    log("checkReturnEligibility", { orderNumber, reason }, output);
    return output;
  },
});
