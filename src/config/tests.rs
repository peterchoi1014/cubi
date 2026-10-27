use super::*;

fn environment(values: &[(&str, &str)]) -> Environment {
    Environment::from_values(
        values
            .iter()
            .map(|(key, value)| ((*key).into(), (*value).into())),
    )
}

fn environment_with_case(values: &[(&str, &str)], case_insensitive: bool) -> Environment {
    Environment::from_values_with_case(
        values
            .iter()
            .map(|(key, value)| ((*key).into(), (*value).into())),
        case_insensitive,
    )
}

#[test]
fn environment_key_lookup_selects_platform_case_semantics() {
    let sensitive = environment_with_case(&[("cubi_model", "lower")], false);
    let insensitive = environment_with_case(&[("cubi_model", "lower")], true);

    assert_eq!(sensitive.value("CUBI_MODEL"), None);
    assert_eq!(insensitive.value("CUBI_MODEL"), Some("lower"));
}

#[test]
fn provider_keys_preserve_original_case_with_windows_semantics() {
    let env = environment_with_case(&[("cubi_openai_api_key", "secret")], true);

    assert_eq!(
        env.provider_keys().get("cubi_openai_api_key"),
        Some(&OsString::from("secret"))
    );
}

#[cfg(windows)]
#[test]
fn captured_windows_environment_accepts_mixed_case_names() {
    let _lock = crate::compat::test_env::env_guard();
    let _restore = crate::compat::test_env::EnvRestore::capture(&["cubi_model"]);
    // SAFETY: the shared environment lock is held through capture and restoration.
    unsafe {
        std::env::set_var("cubi_model", "mixed-case-model");
    }

    let env = Environment::capture();

    assert_eq!(env.value("CUBI_MODEL"), Some("mixed-case-model"));
}

#[test]
fn model_precedence_is_flag_then_env_then_persisted_then_default() {
    let app = AppConfig {
        default_model: Some("saved-model".into()),
        ..AppConfig::default()
    };
    let flags = StartupOverrides {
        model: Some("script-model".into()),
        ..StartupOverrides::default()
    };
    let env = environment(&[("CUBI_MODEL", "shell-model")]);
    assert_eq!(
        RuntimeConfig::resolve(&env, app.clone(), &flags).model,
        "script-model"
    );
    assert_eq!(
        RuntimeConfig::resolve(&env, app.clone(), &StartupOverrides::default()).model,
        "shell-model"
    );
    assert_eq!(
        RuntimeConfig::resolve(&environment(&[]), app, &StartupOverrides::default()).model,
        "saved-model"
    );
    assert_eq!(
        RuntimeConfig::resolve(
            &environment(&[]),
            AppConfig::default(),
            &StartupOverrides::default()
        )
        .model,
        crate::DEFAULT_MODEL
    );
}

#[test]
fn receipts_precedence_preserves_empty_flag_and_skips_empty_env() {
    let app = AppConfig {
        receipts: Some("saved.jsonl".into()),
        ..AppConfig::default()
    };
    let env = environment(&[("CUBI_RECEIPTS", "shell.jsonl")]);
    let flags = StartupOverrides {
        receipts: Some("flag.jsonl".into()),
        ..StartupOverrides::default()
    };
    assert_eq!(
        RuntimeConfig::resolve(&env, app.clone(), &flags)
            .paths
            .receipts,
        Some("flag.jsonl".into())
    );
    assert_eq!(
        RuntimeConfig::resolve(&env, app.clone(), &StartupOverrides::default())
            .paths
            .receipts,
        Some("shell.jsonl".into())
    );
    assert_eq!(
        RuntimeConfig::resolve(
            &environment(&[("CUBI_RECEIPTS", "")]),
            app.clone(),
            &StartupOverrides::default()
        )
        .paths
        .receipts,
        Some("saved.jsonl".into())
    );
    assert_eq!(
        RuntimeConfig::resolve(
            &env,
            app,
            &StartupOverrides {
                receipts: Some(String::new()),
                ..flags
            }
        )
        .paths
        .receipts,
        Some(PathBuf::new())
    );
    assert!(
        RuntimeConfig::resolve(
            &environment(&[]),
            AppConfig::default(),
            &StartupOverrides::default()
        )
        .paths
        .receipts
        .is_none()
    );
}

#[test]
fn persisted_ui_preferences_keep_the_historical_env_precedence_exception() {
    let app = AppConfig {
        theme: Some("light".into()),
        output_style: Some("concise".into()),
        color: Some("off".into()),
        vim_mode: Some("on".into()),
        ..AppConfig::default()
    };
    let env = environment(&[
        ("CUBI_THEME", "dark"),
        ("CUBI_OUTPUT_STYLE", "explanatory"),
        ("CUBI_COLOR", "on"),
        ("CUBI_VIM_MODE", "off"),
    ]);
    let flags = StartupOverrides {
        chat_session: true,
        ..StartupOverrides::default()
    };
    let resolved = RuntimeConfig::resolve(&env, app.clone(), &flags);
    assert_eq!(resolved.ui.theme, "light");
    assert_eq!(resolved.ui.output_style, "concise");
    assert_eq!(resolved.ui.color.as_deref(), Some("off"));
    assert_eq!(resolved.ui.vim_mode, "on");
    let early_command = RuntimeConfig::resolve(&env, app, &StartupOverrides::default());
    assert_eq!(early_command.ui.theme, "dark");
}

#[test]
fn retries_accept_zero_and_do_not_trim_invalid_env() {
    let app = AppConfig {
        llm_max_retries: 7,
        ..AppConfig::default()
    };
    for (input, expected) in [("0", 0), ("4", 4), (" 4 ", 7), ("invalid", 7)] {
        let resolved = RuntimeConfig::resolve(
            &environment(&[("CUBI_LLM_MAX_RETRIES", input)]),
            app.clone(),
            &StartupOverrides::default(),
        );
        assert_eq!(resolved.generation.max_retries, expected);
    }
}

#[test]
fn legacy_promotion_is_pure_and_canonical_presence_wins() {
    let env = environment(&[
        ("AICHAT_THEME", "light"),
        ("AI_CHAT_CLI_MODEL", "old-model"),
        ("CUBI_MODEL", ""),
        ("AICHAT_OPENAI_API_KEY", "old-key"),
        ("CUBI_OPENAI_API_KEY", "new-key"),
    ]);
    let resolved = RuntimeConfig::resolve(&env, AppConfig::default(), &StartupOverrides::default());
    assert_eq!(resolved.ui.theme, "light");
    assert_eq!(resolved.model, crate::DEFAULT_MODEL);
    assert_eq!(
        resolved.provider.keys["CUBI_OPENAI_API_KEY"],
        OsString::from("new-key")
    );
    assert_eq!(env.value("AI_CHAT_CLI_MODEL"), Some("old-model"));
}

#[test]
fn legacy_provider_keys_respect_canonical_presence() {
    for (canonical, expected) in [
        (None, "legacy-key"),
        (Some(""), ""),
        (Some("new-key"), "new-key"),
    ] {
        let mut values = vec![("AICHAT_OPENAI_API_KEY", "legacy-key")];
        if let Some(value) = canonical {
            values.push(("CUBI_OPENAI_API_KEY", value));
        }
        let resolved = RuntimeConfig::resolve(
            &environment(&values),
            AppConfig::default(),
            &StartupOverrides::default(),
        );

        let child = Environment::from_values(resolved.child_env());

        assert_eq!(child.value("CUBI_OPENAI_API_KEY"), Some(expected));
    }
}

#[test]
fn memory_extraction_requires_exact_opt_in() {
    for (value, expected) in [
        (None, false),
        (Some("0"), false),
        (Some("true"), false),
        (Some("1"), true),
    ] {
        let env = Environment::from_values(
            value
                .into_iter()
                .map(|value| ("CUBI_COMPACT_EXTRACT_MEMORIES".into(), value.into())),
        );

        let resolved =
            RuntimeConfig::resolve(&env, AppConfig::default(), &StartupOverrides::default());

        assert_eq!(resolved.compact_extract_memories, expected);
    }
}

#[test]
fn generation_values_and_fake_hooks_round_trip_to_child_configuration() {
    let env = environment(&[
        ("CUBI_NUM_CTX", "8192"),
        ("CUBI_CONTEXT_WINDOW", "16384"),
        ("CUBI_MAX_TOKENS", "33"),
        ("CUBI_THINK", "OFF"),
        ("CUBI_KEEP_ALIVE", " 5m "),
        ("CUBI_HTTP_TIMEOUT_SECS", "17"),
        ("CUBI_TEMPERATURE", "0.2"),
        ("CUBI_TOP_P", "0.8"),
        ("CUBI_TOP_K", "9"),
        ("CUBI_TOOL_PROFILE", "coding"),
        ("CUBI_READ_FILE_MAX_LINES", "123"),
        ("CUBI_READ_FILE_MAX_BYTES", "4567"),
        ("CUBI_COMPACT_EXTRACT_MEMORIES", "1"),
        ("CUBI_FAKE_LLM", ""),
        ("CUBI_FAKE_LLM_RESPONSE", "fixture"),
        ("CUBI_FAKE_LLM_MODEL_RESPONSES", r#"{"model":"specific"}"#),
        (
            "CUBI_FAKE_LLM_TOOL_CALL",
            r#"{"function":{"name":"read_file","arguments":{"path":"x"}}}"#,
        ),
        ("CUBI_FAKE_LLM_TOOL_CALL_REPEAT", "true"),
        ("CUBI_FAKE_LLM_FAIL_MODELS", "bad, worse"),
    ]);
    let parent = RuntimeConfig::resolve(&env, AppConfig::default(), &StartupOverrides::default());
    let child_env = Environment::from_values(parent.child_env());
    let child = RuntimeConfig::resolve(
        &child_env,
        AppConfig::default(),
        &StartupOverrides::default(),
    );
    assert_eq!(child.generation.num_ctx, Some(8192));
    assert_eq!(child.generation.context_window, Some(16384));
    assert_eq!(child.generation.max_tokens, Some(33));
    assert_eq!(child.generation.think, Some(false));
    assert_eq!(child.generation.keep_alive.as_deref(), Some("5m"));
    assert_eq!(child.generation.http_timeout, Duration::from_secs(17));
    let sampling = child.generation.sampling_for_model("unknown").unwrap();
    assert_eq!(sampling.temperature, Some(0.2));
    assert_eq!(sampling.top_p, Some(0.8));
    assert_eq!(sampling.top_k, Some(9));
    assert_eq!(child.tool_profile, ToolProfile::Coding);
    assert_eq!(
        (child.read_file_max_lines, child.read_file_max_bytes),
        (123, 4567)
    );
    assert!(child.compact_extract_memories);
    assert!(child.fake.enabled && child.fake.tool_call_repeat);
    assert_eq!(child.fake.response, "fixture");
    assert_eq!(child.fake.model_responses["model"], "specific");
    assert_eq!(child.fake.tool_call, parent.fake.tool_call);
    assert_eq!(child.fake.fail_models, ["bad", "worse"]);
}

#[test]
fn invalid_sampling_is_deferred_and_preserved_for_children() {
    let parent = RuntimeConfig::resolve(
        &environment(&[("CUBI_TOP_P", "NaN")]),
        AppConfig::default(),
        &StartupOverrides::default(),
    );
    let child = RuntimeConfig::resolve(
        &Environment::from_values(parent.child_env()),
        AppConfig::default(),
        &StartupOverrides::default(),
    );
    assert_eq!(
        child
            .generation
            .sampling_for_model("bonsai")
            .unwrap_err()
            .to_string(),
        "CUBI_TOP_P must be a finite number in (0, 1]"
    );
}

#[test]
fn child_configuration_preserves_log_filter() {
    let parent = RuntimeConfig::resolve(
        &environment(&[("CUBI_LOG", "cubi=debug")]),
        AppConfig::default(),
        &StartupOverrides::default(),
    );

    let child = Environment::from_values(parent.child_env());

    assert_eq!(child.value("CUBI_LOG"), Some("cubi=debug"));
}
