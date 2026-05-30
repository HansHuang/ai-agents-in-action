import { spawnSync } from "node:child_process";

const mode = process.argv[2] ?? "template";

if (mode === "template") {
	const { main } = await import("./prompt_template.js");
	main().catch(console.error);
} else if (mode === "few-shot") {
	const result = spawnSync("npx", ["tsx", "few_shot_comparison.ts"], {
		stdio: "inherit",
	});
	process.exitCode = result.status ?? 1;
} else if (mode === "chain-of-thought") {
	const result = spawnSync("npx", ["tsx", "chain_of_thought.ts"], {
		stdio: "inherit",
	});
	process.exitCode = result.status ?? 1;
} else {
	console.error(
		`Unknown mode "${mode}". Use: template, few-shot, or chain-of-thought.`
	);
	process.exitCode = 1;
}
