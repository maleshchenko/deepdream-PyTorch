<img width="300" height="176" alt="deepdream_inception_v3_gradient_step" src="https://github.com/user-attachments/assets/767d4e7e-8f55-4153-8b79-ad56085a9cac" /># DeepDream in PyTorch

DeepDream is a computer vision algorithm that produces psychedelic, dreamlike images by enhancing patterns in existing images using a deep neural network. It works by performing gradient ascent in feature space (as opposed to gradient descent in typical machine learning), causing the network to amplify patterns it recognizes. This helps visualize what features a deep network has learned, while producing visually striking artistic results.

DeepDream was invented by Google and applied to the Inception network (trained on ImageNet) in 2014. The original algorithm was implemented in Caffe. This repository is a **PyTorch implementation** using InceptionV3, making it easy to run and understand the core concepts behind the algorithm.

### Example:

```python test_pytorch.py --content source.jpg --layer Mixed_5b --iterations 200 --step 1.6 --octaves 5```

<img width="400" height="400" alt="source" src="https://github.com/user-attachments/assets/e755cea5-75aa-4e21-a640-8c57aa7985dc" /> <img width="400" height="400" alt="deepdream_pytorch" src="https://github.com/user-attachments/assets/ff33b51e-02fc-480c-8667-dc0be42db1dd" />

## How It Works

1. Load a pre-trained neural network (InceptionV3) trained on ImageNet
2. Start with an input image (or random noise)
3. Forward pass through the network to extract activations from selected layers
4. Compute gradients with respect to the input image to maximize activations
5. Update the image in the direction that enhances detected patterns
6. Repeat to create increasingly "dreamy" results

![Uploading deep<svg width="100%" viewBox="0 0 680 400" role="img" style="" xmlns="http://www.w3.org/2000/svg" xmlns:c2pa="http://c2pa.org/manifest"><metadata><c2pa:manifest>AAAWkGp1bWIAAAAeanVtZGMycGEAEQAQgAAAqgA4m3EDYzJwYQAAABZqanVtYgAAAEdqdW1kYzJtYQARABCAAACqADibcQN1cm46YzJwYTo5OTgyOWIzYS0wMTg2LTQ5NzItYTJhNC0xNjQ2MWUwY2UyZTUAAAADf2p1bWIAAAApanVtZGMyYXMAEQAQgAAAqgA4m3EDYzJwYS5hc3NlcnRpb25zAAAAALxqdW1iAAAARGp1bWRjYm9yABEAEIAAAKoAOJtxE2MycGEuaW5ncmVkaWVudC52MwAAAAAYYzJzaGTJQt93dRd1rE4BOwSDYR0AAABwY2JvcqNscmVsYXRpb25zaGlwaHBhcmVudE9maWRjOmZvcm1hdG1pbWFnZS9zdmcreG1samluc3RhbmNlSUR4LHhtcDppaWQ6MTk4MGYzMzQtMTAwNS00NjliLWI1YzMtNzkwMjZlMzRiMmQ4AAABzmp1bWIAAABBanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5hY3Rpb25zLnYyAAAAABhjMnNoXNOy/WwQ+qfLOHGhgib9YQAAAYVjYm9yoWdhY3Rpb25zgqJmYWN0aW9ua2MycGEub3BlbmVkanBhcmFtZXRlcnOha2luZ3JlZGllbnRzgaJjdXJseC1zZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmluZ3JlZGllbnQudjNkaGFzaFggjgSKsteLofGdwTwVpvhV3+VzNjTda2rJm2peDNhFzFSkZmFjdGlvbngdY29tLmFudGhyb3BpYy5jbGF1ZGUucHJvdmlkZWRtc29mdHdhcmVBZ2VudKFkbmFtZWZDbGF1ZGVqcGFyYW1ldGVyc6F4H2NvbS5hbnRocm9waWMub3JpZ2luLWNvbmZpZGVuY2VndW5rbm93bmtkZXNjcmlwdGlvbnhmQ2xhdWRlIHByb3ZpZGVkIHRoaXMgZmlsZSBhdCB0aGUgcmVxdWVzdCBvZiBhIHVzZXIgYW5kIG1heSBoYXZlIGNyZWF0ZWQgb3IgbW9kaWZpZWQgdGhlIGZpbGUgY29udGVudHMuAAAAxGp1bWIAAABAanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5oYXNoLmRhdGEAAAAAGGMyc2h6NCH33VWr+l/msZ21U5InAAAAfGNib3KlamV4Y2x1c2lvbnOBomVzdGFydBieZmxlbmd0aBkeGGRuYW1lbmp1bWJmIG1hbmlmZXN0Y2FsZ2ZzaGEyNTZkaGFzaFggCoMgom970h5vSRZwl7JOeHt0RcVbe6Vd/7duuBIjFX9jcGFkSQAAAAAAAAAAAAAAAmRqdW1iAAAAJ2p1bWRjMmNsABEAEIAAAKoAOJtxA2MycGEuY2xhaW0udjIAAAACNWNib3Kmamluc3RhbmNlSUR4LHhtcDppaWQ6ZDcwMjFmYzYtZDNlZi00MWRlLTk0MzAtYTk5NjIyM2ZhNDEwdGNsYWltX2dlbmVyYXRvcl9pbmZvo2RuYW1lc0FudGhyb3BpYyBDbGF1ZGUuYWlndmVyc2lvbmUxLjAuMHdvcmcuY29udGVudGF1dGguYzJwYV9yc2YwLjkwLjBpc2lnbmF0dXJleE1zZWxmI2p1bWJmPS9jMnBhL3VybjpjMnBhOjk5ODI5YjNhLTAxODYtNDk3Mi1hMmE0LTE2NDYxZTBjZTJlNS9jMnBhLnNpZ25hdHVyZXJjcmVhdGVkX2Fzc2VydGlvbnOComN1cmx4KnNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2MycGEuYWN0aW9ucy52MmRoYXNoWCBPR5vdfe07coa7ujJhuioBe2865bVcNv5xzQnsmR40t6JjdXJseClzZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmhhc2guZGF0YWRoYXNoWCAaRrHkYJmGAVy3D8iTA1NGSj7KbI7Jwo4Yaarp8qySCnNnYXRoZXJlZF9hc3NlcnRpb25zgaJjdXJseC1zZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmluZ3JlZGllbnQudjNkaGFzaFggjgSKsteLofGdwTwVpvhV3+VzNjTda2rJm2peDNhFzFRjYWxnZnNoYTI1NgAAEDhqdW1iAAAAKGp1bWRjMmNzABEAEIAAAKoAOJtxA2MycGEuc2lnbmF0dXJlAAAAEAhjYm9y0oRZAhKiASYYIVkCCjCCAgYwggGNoAMCAQICFEDloAruwjnQvriD+gZCBT1nVRMAMAoGCCqGSM49BAMDMEkxFzAVBgNVBAoTDkFudGhyb3BpYywgUEJDMS4wLAYDVQQDEyVBbnRocm9waWMgQ29udGVudCBDcmVkZW50aWFscyBSb290IENBMB4XDTI2MDgwNzE4NDM1NloXDTI4MDgwNjE5NDM1NlowRDEXMBUGA1UEChMOQW50aHJvcGljLCBQQkMxKTAnBgNVBAMTIEFudGhyb3BpYyBDbGF1ZGUgQ29udGVudCBTaWduaW5nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEmHoKa8tQGAUU1TS9QqU5W0Tp2N3XsvlK7BfQt6YWKwEzd2R3/dzKPEUDdCjlLjp9fT+KFjRVnuZ9v0oXvTe3k6NYMFYwDgYDVR0PAQH/BAQDAgeAMBUGA1UdJQQOMAwGCisGAQQBg+heAgEwDAYDVR0TAQH/BAIwADAfBgNVHSMEGDAWgBTOUeIEgU5kWyP448TPmj6cwddcwjAKBggqhkjOPQQDAwNnADBkAjAxcx0UngF60stVjs5G4T2eiptsBk5mf9oCtfJPAUBl8qs/PEXa8+gk1/X5QJ2DVcYCMHBfXN31YapiSqYvlIWrDVDJKOvXMl+kkz37Wt0PBI8sw486Mq6JeOhT+lRR4b1HCaFjcGFkWQ2eAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9lhA62Yq90g60VvC2hAbvihLIwwykJmpH020muXtUbA2eKHlHf5Wjrhu15k3RzrBmR/WW2tc3/JrGDfNYPRjTvPkGw==</c2pa:manifest></metadata><title style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">DeepDream gradient step through InceptionV3</title><desc style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">An image tile enters the InceptionV3 stem and Mixed blocks. A forward hook on one Mixed block feeds a mean-activation loss. Backward gives pixel gradients, which update the image and repeat.</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="stroke:#898781;stroke:context-stroke;"/></marker></defs>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="40" y="50" width="104" height="56" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="92" y="68" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Stem</text><text x="92" y="88" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">conv + pool</text></g>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="164" y="50" width="104" height="56" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="216" y="68" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Mixed_5</text><text x="216" y="88" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">5b, 5c, 5d</text></g>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="288" y="50" width="104" height="56" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="340" y="68" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Mixed_6</text><text x="340" y="88" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">6a to 6e</text></g>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="412" y="50" width="104" height="56" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="464" y="68" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Mixed_7</text><text x="464" y="88" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">7a, 7b, 7c</text></g>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="536" y="50" width="104" height="56" rx="8" stroke-width="0.5" stroke-dasharray="4 3" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-dasharray:4px, 3px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="588" y="68" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Pool + fc</text><text x="588" y="88" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">still computed</text></g>
<line x1="146" y1="78" x2="162" y2="78" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="270" y1="78" x2="286" y2="78" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="394" y1="78" x2="410" y2="78" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="518" y1="78" x2="534" y2="78" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="40" y="180" width="120" height="56" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="100" y="198" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Random tile</text><text x="100" y="218" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">roll + crop</text></g>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="275" y="180" width="130" height="56" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="340" y="198" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Loss</text><text x="340" y="218" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">mean activation</text></g>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="520" y="180" width="120" height="56" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="580" y="198" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Gradient</text><text x="580" y="218" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">pixel grads</text></g>
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="510" y="300" width="130" height="56" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/><text x="575" y="318" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Update image</text><text x="575" y="338" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">g / std × step</text></g>
<line x1="92" y1="178" x2="92" y2="108" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="102" y="146" dominant-baseline="central" style="fill:rgb(82, 81, 78);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">input tile</text>
<line x1="340" y1="108" x2="340" y2="178" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="350" y="146" dominant-baseline="central" style="fill:rgb(82, 81, 78);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">forward hook</text>
<line x1="407" y1="212" x2="518" y2="212" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="462" y="198" text-anchor="middle" dominant-baseline="central" style="fill:rgb(82, 81, 78);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">backward</text>
<line x1="580" y1="238" x2="580" y2="298" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<path d="M508 328 L100 328 L100 240" fill="none" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="300" y="312" text-anchor="middle" dominant-baseline="central" style="fill:rgb(82, 81, 78);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:ui-sans-serif, system-ui, -apple-system, &quot;Segoe UI&quot;, Roboto, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">repeat per iteration and octave</text>
</svg>dream_inception_v3_gradient_step.svg…]()

## Requirements

- Python >= 3.13
- PyTorch >= 2.0.0
- PyTorchVision >= 0.15.0
- NumPy, Pillow, SciPy (for image processing)
- Matplotlib (optional, for visualization)

## Installation

### 1. Create a Virtual Environment

```bash
# Using venv (built-in to Python)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda (if you prefer)
conda create -n deepdream python=3.13
conda activate deepdream
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install Development Dependencies

For visualization support with Matplotlib:
```bash
pip install -e ".[dev]"
```

**Note:** The project does not rely on any bundled environments. Each user should create their own virtual environment as shown above.

## Usage

### Basic Usage (Random Noise)

Generate DeepDream from scratch using random noise:

```bash
python test_pytorch.py
```

### Apply to an Existing Image

```bash
python test_pytorch.py --content my_image.jpg --output result.jpg
```

### Advanced Options

```bash
python test_pytorch.py \
    --content input.jpg \
    --output output.jpg \
    --layer Mixed_7c \
    --iterations 30 \
    --step 0.8 \
    --octaves 4
```

## Parameters

- **--content**: Source image file (if not provided, generates from random noise)
- **--output**: Output filename for the generated image
- **--layer**: Which layer to target for activation maximization
  - Deeper layers (Mixed_7x): More complex, semantic features
  - Middle layers (Mixed_5x, Mixed_6x): Medium-level patterns
  - Default: `Mixed_5c`
- **--iterations**: How many gradient ascent steps per octave (higher = more detailed)
- **--step**: Gradient step size - controls how aggressively patterns are enhanced (0.1-2.0)
- **--octaves**: Number of scales to process - higher octaves enhance larger patterns first

## Available Layers

```
Mixed_5b, Mixed_5c, Mixed_6a, Mixed_6b, Mixed_6c, Mixed_7a, Mixed_7b, Mixed_7c
```

**Layer Selection Tips:**
- `Mixed_5b/5c`: Good for delicate, intricate patterns
- `Mixed_6b/6c`: Balanced results with both textures and objects
- `Mixed_7a/7b`: Strong object-like features, more pronounced patterns

## Algorithm Variants

The code includes two rendering algorithms:

### 1. Multi-Scale Octave Processing (Default - `render_deepdream`)
Highest quality. Processes image from coarse to fine resolution.
- **Pros**: Better results, captures patterns at multiple scales
- **Cons**: Slower, uses more memory

### 2. Naive Gradient Ascent (`render_naive`)
Simple single-scale approach.
- **Pros**: Fast, lower memory usage
- **Cons**: Lower quality results, can produce artifacts

## Technical Details

### Multi-Scale Octave Processing

The algorithm uses a Laplacian pyramid approach:

1. **Decomposition**: Split image into Laplacian bands (coarse → fine)
2. **Coarse-to-Fine Processing**: Start with low resolution, progressively add details
3. **Reconstruction**: Combine processed bands back into final image

This prevents small-scale artifacts from overwhelming the result and enables coherent pattern enhancement across scales.

### Gradient Tiling

To handle large images without GPU memory issues:
- Image is processed in overlapping tiles (512x512)
- Random shifts applied to reduce tile boundary artifacts
- Gradients accumulated across all tiles

### Normalization

Gradients are normalized to prevent explosion:
```
gradient = gradient / (std(gradient) + ε)
```

This ensures stable updates regardless of gradient magnitude.

## Output Files

The script generates:
- `original_image_pytorch.jpg`: Input image (for reference)
- `deepdream_pytorch.jpg`: Final DeepDream result (multi-scale)

## Examples

### Artistic Enhancement
```bash
python test_pytorch.py --content source.jpg --layer Mixed_6c \
    --iterations 25 --step 0.6 --octaves 5
```

### Subtle Effects (Lower Iterations)
```bash
python test_pytorch.py --content source.jpg --layer Mixed_5b \
    --iterations 10 --step 0.3 --octaves 2
```

### Extreme Psychedelic Effects
```bash
python test_pytorch.py --content source.jpg --layer Mixed_7c \
    --iterations 50 --step 1.5 --octaves 6
```

## Docker

The image uses CPU PyTorch by default. Mount the current directory at `/data` so input images and outputs stay on the host.

### Build

```bash
docker build -t deepdream-pytorch .
```

GPU build (Linux with NVIDIA Container Toolkit):

```bash
docker build -t deepdream-pytorch --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .
```

### Run

```bash
# Random noise → deepdream_pytorch.jpg and original_image_pytorch.jpg in the current directory
docker run --rm -v "$(pwd):/data" deepdream-pytorch

# Existing image
docker run --rm -v "$(pwd):/data" deepdream-pytorch --content my_image.jpg --output result.jpg

# Extra options (same flags as local usage)
docker run --rm -v "$(pwd):/data" deepdream-pytorch \
    --content input.jpg \
    --output output.jpg \
    --layer Mixed_7c \
    --iterations 30 \
    --step 0.8 \
    --octaves 4
```

GPU run (after a CUDA image build):

```bash
docker run --rm --gpus all -v "$(pwd):/data" deepdream-pytorch --content my_image.jpg --output result.jpg
```

### Docker Compose

```bash
docker compose build
docker compose run --rm deepdream
docker compose run --rm deepdream --content my_image.jpg --output result.jpg
```

For GPU, set `gpus: all` in `docker-compose.yml` and rebuild with `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124`.


## Tips for Best Results

1. **Start with good source images**: Interesting textures and patterns work well
2. **Try different layers**: Each layer produces different visual styles
3. **Adjust iterations**: More iterations = more pronounced effects
4. **Use appropriate step size**: Too high causes artifacts, too low gives subtle effects
5. **Multiple octaves**: Usually 3-5 octaves gives best results
6. **GPU recommended**: CUDA GPU significantly speeds up processing

## Hardware Considerations

- **GPU Memory**: Full HD images need ~8GB VRAM
- **CPU Mode**: Works but very slow (10-50x slower than GPU)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM for best performance

## References

- [Original DeepDream by Google](https://github.com/google/deepdream)
- [Original Implementation in Caffe](https://github.com/google/deepdream/blob/master/dream.ipynb)
- [TensorFlow Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)
- [Wikipedia on DeepDream](https://en.wikipedia.org/wiki/DeepDream)
- [InceptionV3 Architecture](https://arxiv.org/abs/1512.00567)
- [Deep Inside Convolutional Networks: Visualising Image Classification Models](https://arxiv.org/abs/1312.6034)


## References
* [Original Implementation in Caffe](https://github.com/google/deepdream/blob/master/dream.ipynb)
* [TensorFlow Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb) (what I followed)
* [Wikipedia on DeepDream](https://en.wikipedia.org/wiki/DeepDream)
* [Inception TensorFlow Code](https://github.com/tensorflow/models/tree/master/inception) (the network commonly used in the algorithm)
* [A Nice Explanation of DeepDream](http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/)
* [A Nice Video on DeepDream](https://www.youtube.com/watch?v=MrBzgvUNr4w) (moves pretty fast when it gets to the code part, be warned)

