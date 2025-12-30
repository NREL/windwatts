import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import { globalIgnores } from "eslint/config";
import prettier from "eslint-config-prettier";
import importPlugin from "eslint-plugin-import";

export default tseslint.config(
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      importPlugin.flatConfigs.recommended,
      importPlugin.flatConfigs.typescript,
      reactHooks.configs["recommended-latest"],
      reactRefresh.configs.vite,
      prettier,
    ],
    languageOptions: {
      ecmaVersion: 2023,
      globals: globals.browser,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      // Enforce named exports for consistency and maintainability
      "import/no-default-export": "error",
      // Disallow spaces inside single-line blocks: () => {} not () => { }
      "block-spacing": ["error", "never"],
    },
  },
  {
    files: ["vite.config.{ts,js}"],
    rules: {
      "import/no-default-export": "off",
    },
  }
);
